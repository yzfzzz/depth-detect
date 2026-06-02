#include "motion_state_engine.h"

#include "logger_manager.h"

#include <cstdio>
#include <iostream>
#include <opencv2/core/operations.hpp>

MotionStateEngine::MotionStateEngine(float velocity_threshold,
                                     float acceleration_threshold,
                                     float kf_process_noise_cov,
                                     float kf_measurement_noise_cov) :

    velocity_threshold_(velocity_threshold),
    acceleration_threshold_(acceleration_threshold),
    kf_process_noise_cov_(kf_process_noise_cov),
    kf_measurement_noise_cov_(kf_measurement_noise_cov) {}

MotionStateInfoRecord MotionStateEngine::computeMotionState(int    track_id,
                                                            float  raw_value,
                                                            double timestamp) {
    if (raw_value <= 0.0f) {
        return MotionStateInfoRecord(MotionState::INVAILD, MotionState::INVAILD, 0.0f);
    }

    // 1. 获取或创建对应 track_id 的滤波状态
    auto & state = kf_states_[track_id];

    // ================== 卡尔曼滤波初始化 ==================
    if (!state.is_initialized) {
        // 状态维度 3: [值, 速度, 加速度]^T
        // 测量维度 1: [观测到的值（视差/深度）]
        state.kf.init(3, 1, 0);

        // 初始化状态转移矩阵 F  (在预测时会根据 dt 更新)
        // x_k = x_{k-1} + v*dt + 0.5*a*dt^2
        // v_k = v_{k-1} + a*dt
        // a_k = a_{k-1}
        cv::setIdentity(state.kf.transitionMatrix);

        // 测量矩阵 H (我们只测量到了第一个元素)
        state.kf.measurementMatrix                 = cv::Mat::zeros(1, 3, CV_32F);
        state.kf.measurementMatrix.at<float>(0, 0) = 1.0f;

        // 过程噪声协方差矩阵 Q
        // (决定系统的平滑度，值越小越平滑但响应越慢，值越大越灵敏但抗噪弱)
        // [由于加速度本身也是会变的，这里可以设置小一点]
        cv::setIdentity(state.kf.processNoiseCov, cv::Scalar::all(kf_process_noise_cov_));

        // 测量噪声协方差矩阵 R
        // (决定对当前传入雷达/双目数值的信任度，测量噪声大则增大此值)
        cv::setIdentity(state.kf.measurementNoiseCov, cv::Scalar::all(kf_measurement_noise_cov_));

        // 误差协方差矩阵 P (初始的置信度，随便设个稍微大点的值)
        cv::setIdentity(state.kf.errorCovPost, cv::Scalar::all(1));

        // 状态初始化
        state.kf.statePost   = (cv::Mat_<float>(3, 1) << raw_value, 0.0f, 0.0f);
        state.last_timestamp = timestamp;
        state.is_initialized = true;

        return MotionStateInfoRecord(MotionState::STABLE, MotionState::CONSTANT, 0.0f);
    }

    // ================== 卡尔曼滤波预测与更新 ==================
    float dt = static_cast<float>(timestamp - state.last_timestamp);
    if (dt <= 0.0f) {
        dt = 0.033f;  // 兜底保护，假设默认30fps
    }

    // 动态更新状态转移矩阵 (根据 dt)
    state.kf.transitionMatrix.at<float>(0, 1) = dt;
    state.kf.transitionMatrix.at<float>(0, 2) = 0.5f * dt * dt;
    state.kf.transitionMatrix.at<float>(1, 2) = dt;

    // 1. 预测 (Predict)
    state.kf.predict();

    // 2. 更新 (Correct) 融入当前观测值
    cv::Mat measurement     = (cv::Mat_<float>(1, 1) << raw_value);
    cv::Mat estimated_state = state.kf.correct(measurement);

    // 获取滤波后的最优状态
    float smoothed_value   = estimated_state.at<float>(0, 0);
    float current_velocity = estimated_state.at<float>(1, 0);
    float current_accel    = estimated_state.at<float>(2, 0);

    state.last_timestamp = timestamp;  // 记录本帧时间供下一帧用

    // ================== 状态判定 ==================
    // 注意：如果是视差 (Disparity)，物体靠近 => 视差变大 => 速度应为 正数 (>0)
    // 如果是确切深度 (Depth)，物体靠近 => 深度变小 => 速度应为 负数 (<0)

    MotionState direction_state = MotionState::STABLE;
    MotionState accel_state     = MotionState::CONSTANT;

    // 此处假设为视差逻辑 (值变大=靠近)
    if (current_velocity > velocity_threshold_) {
        direction_state = MotionState::APPROACH;
        if (current_accel > acceleration_threshold_) {
            accel_state = MotionState::ACCELE;
        } else if (current_accel < -acceleration_threshold_) {
            accel_state = MotionState::DECELE;
        }
    }
    // 视差变小=远离
    else if (current_velocity < -velocity_threshold_) {
        direction_state = MotionState::MOVE_AWAY;
        if (current_accel < -acceleration_threshold_) {
            accel_state = MotionState::ACCELE;  // 远离时加速跑
        } else if (current_accel > acceleration_threshold_) {
            accel_state = MotionState::DECELE;
        }
    }

    return MotionStateInfoRecord(direction_state, accel_state, current_velocity);
}

float MotionStateEngine::getObjectDepth(cv::Mat depth, const STrack & track, cv::Size image_size) {
    if (!depth.empty()) {
        cv::resize(depth, depth, image_size);
    } else {
        APP_WARN("depth_map is empty!");
        return 0.0f;
    }

    const std::vector<float> & tlwh        = track.tlwh_;
    float                      depth_value = 0.0f;

    depth_value = computeMeanDepth(depth, tlwh);

    return depth_value;
}

float MotionStateEngine::computeMeanDepth(cv::Mat                    depth,
                                          const std::vector<float> & tlwh,
                                          int                        num_samples) const {
    int left   = static_cast<int>(tlwh[0]);
    int top    = static_cast<int>(tlwh[1]);
    int right  = static_cast<int>(tlwh[0] + tlwh[2]);
    int bottom = static_cast<int>(tlwh[1] + tlwh[3]);

    left   = std::max(0, std::min(left, depth.cols - 1));
    top    = std::max(0, std::min(top, depth.rows - 1));
    right  = std::max(0, std::min(right, depth.cols - 1));
    bottom = std::max(0, std::min(bottom, depth.rows - 1));

    int width  = right - left;
    int height = bottom - top;

    if (width <= 0 || height <= 0) {
        return 0.0f;
    }

    // 存储当前目标收集到的有效深度点
    std::vector<float> sampled_depths;

    int   grid_size = static_cast<int>(std::sqrt(num_samples));
    float step_x    = static_cast<float>(width) / std::max(1, grid_size - 1);
    float step_y    = static_cast<float>(height) / std::max(1, grid_size - 1);

    // 提前构建需要排除的遮挡区域（假设交并面积大且目前只做简单的框剔除）
    // 为了防止互相剔除，我们需要大致知道谁在前谁在后。
    // 但是这里我们用一个简单粗暴的方法：只要这个像素落在了任何其他 bbox 内，
    // 我们在这个提取阶段暂时不能武断地全剔除（因为可能它是被检测错了），
    // 所以这里的优化着重在第 2 步的统计过滤。但为了减少影响，如果中心点靠近边缘的，你可以剔除。

    // 重点优化：网格区域智能采样
    // 1. 尽量往目标框的核心（中心）区域聚集采样，因为边缘更有可能是背景遮挡
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // 这里可以在 i, j 循环里可以加一个高斯权重或者边界缩收
            // 比如只采样框的中心 60% 区域：
            float shrink_ratio = 0.2f;  // 上下左右各缩进20%
            int   x            = left + static_cast<int>(width * shrink_ratio) +
                    static_cast<int>(i * (width * (1.0f - 2 * shrink_ratio)) /
                                     std::max(1, grid_size - 1));
            int y = top + static_cast<int>(height * shrink_ratio) +
                    static_cast<int>(j * (height * (1.0f - 2 * shrink_ratio)) /
                                     std::max(1, grid_size - 1));

            if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows) {
                continue;
            }

            float depth_value = 0.0f;
            if (depth.type() == CV_32FC1) {
                depth_value = depth.at<float>(y, x);
            } else if (depth.type() == CV_8UC1) {
                depth_value = static_cast<float>(depth.at<uchar>(y, x));
            }

            if (depth_value > 0.01f) {
                sampled_depths.push_back(depth_value);
            }
        }
    }

    if (sampled_depths.empty()) {
        return 0.0f;
    }

    // 重点优化：基于统计学的鲁棒均值选取（截断均值法 Truncated Mean / 一维中值聚类）
    // 对收集到的所有像素点进行排序
    std::sort(sampled_depths.begin(), sampled_depths.end());

    // 在一个框里，背景的深度值一定远大于前景的目标值。如果该目标框是被遮挡在后面的，
    // 那么被遮挡到的那部分像素值一定是非常小（前景）的。
    // 如果框本身偏大，框进去了后面的背景，那部分像素值一定非常大。
    // 因此，如果是为了获取**本物体**最真实的深度，需要剔除两头：
    // 极小值（可能是挡在它前面的物体）；极大值（可能是穿透过去打在远处墙上的深度）。

    // 因此，我们计算去除掉最小的 25% (可能的前景遮挡) 和最大的 25% (背景透视) 后的均值
    int num_valid = sampled_depths.size();
    if (num_valid < 4) {
        // 数据太少，直接取中位数
        return sampled_depths[num_valid / 2];
    }

    int skip_low = static_cast<int>(num_valid * 0.25f);  // 剔除25%最近距离（前景毛刺与遮挡）
    int skip_high = static_cast<int>(num_valid * 0.25f);  // 剔除25%最远距离（背景噪声）

    float sum   = 0.0f;
    int   count = 0;
    for (int i = skip_low; i < num_valid - skip_high; ++i) {
        sum += sampled_depths[i];
        count++;
    }

    return count > 0 ? (sum / count) : 0.0f;
}
