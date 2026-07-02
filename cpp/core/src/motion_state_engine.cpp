#include "motion_state_engine.h"

#include "logger_manager.h"

#include <cstdio>
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

    // 卡尔曼滤波初始化：状态维度=3 [位置,速度,加速度]，测量维度=1（仅观测位置）
    if (!state.is_initialized) {
        // 状态转移矩阵 F 在预测时根据 dt 动态更新
        // x_k = x_{k-1} + v*dt + 0.5*a*dt^2, v_k = v_{k-1} + a*dt, a_k = a_{k-1}
        state.kf.init(3, 1, 0);

        // 测量矩阵 H - 仅测量位置（第一个元素）
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

    // 卡尔曼滤波预测与更新：根据时间间隔 dt 更新状态转移矩阵
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

    state.last_timestamp = timestamp;

    // 运动状态判定：基于卡尔曼滤波估算的速度和加速度
    // 注意：此逻辑基于视差（值变大=物体靠近），若使用绝对深度则需要反转方向判断

    MotionState direction_state = MotionState::STABLE;
    MotionState accel_state     = MotionState::CONSTANT;

    // 当前按视差逻辑处理：值变大 → 靠近，若使用深度则需反转符号
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
            accel_state = MotionState::ACCELE;  // 远离且加速远离（加速度与速度同向）
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

    // 在目标框内均匀网格采样，统计有效深度值
    // 采样策略：缩进 20% 边界以避开边缘背景，按 grid_size × grid_size 在框内均匀采点
    int   grid_size    = static_cast<int>(std::sqrt(num_samples));
    float shrink_ratio = 0.2f;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int x = left + static_cast<int>(width * shrink_ratio) +
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

    // 截断均值法：先排序，剔除两端 25% 异常值（前 25% 可能是前景遮挡，后 25% 可能是背景噪声）
    // 再对中间 50% 的数据取均值，得到该目标在当前帧的鲁棒深度估计
    std::sort(sampled_depths.begin(), sampled_depths.end());
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
