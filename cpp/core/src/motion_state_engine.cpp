#include "motion_state_engine.h"

#include "logger_manager.h"

#include <cstdio>
#include <opencv2/core/operations.hpp>

MotionStateEngine::MotionStateEngine(float velocity_threshold,
                                     float acceleration_threshold,
                                     float velocity_hysteresis,
                                     float acceleration_hysteresis,
                                     float kf_process_noise_cov,
                                     float kf_measurement_noise_cov,
                                     float min_scale_for_ttc,
                                     float min_velocity_for_ttc,
                                     float ema_alpha,
                                     float bbox_jump_ratio_threshold,
                                     float ttc_warn_threshold,
                                     float ttc_clear_threshold,
                                     int   ttc_enter_frames,
                                     int   ttc_exit_frames) :
    velocity_threshold_(velocity_threshold),
    acceleration_threshold_(acceleration_threshold),
    velocity_hysteresis_(std::max(0.0f, velocity_hysteresis)),
    acceleration_hysteresis_(std::max(0.0f, acceleration_hysteresis)),
    kf_process_noise_cov_(kf_process_noise_cov),
    kf_measurement_noise_cov_(kf_measurement_noise_cov),
    min_scale_for_ttc_(min_scale_for_ttc),
    min_velocity_for_ttc_(std::max(0.0f, min_velocity_for_ttc)),
    ema_alpha_(ema_alpha),
    bbox_jump_ratio_threshold_(bbox_jump_ratio_threshold),
    ttc_warn_threshold_(std::max(0.0f, ttc_warn_threshold)),
    ttc_clear_threshold_(std::max(ttc_warn_threshold_, std::max(0.0f, ttc_clear_threshold))),
    ttc_enter_frames_(std::max(1, ttc_enter_frames)),
    ttc_exit_frames_(std::max(1, ttc_exit_frames)) {}

MotionStateInfoRecord MotionStateEngine::computeMotionStateFromDepth(int    track_id,
                                                                     float  raw_depth,
                                                                     double timestamp) {
    const auto filtered = computeStateImpl(track_id, raw_depth, timestamp);
    if (!filtered.valid) {
        return MotionStateInfoRecord(MotionState::INVAILD, MotionState::INVAILD, 0.0f);
    }

    MotionState direction   = MotionState::STABLE;
    MotionState accel_state = MotionState::CONSTANT;
    float       ttc         = -1.0f;
    bool        ttc_danger  = false;

    if (!filtered.first_frame) {
        // 深度模型输出可能是视差（近大远小）或深度（近小远大），符号约定不定，
        // 状态判定与 TTC 均按 |速度| 处理，与符号无关
        determineMotionStates(track_id, filtered.velocity, filtered.acceleration, direction,
                              accel_state);

        // TTC = 深度值 / |速度|；跳变帧拒绝，相对变化率过小（TTC 巨大）或值近 0 视为无效
        const float speed = std::abs(filtered.velocity);
        if (!filtered.is_large_jump && speed > 1e-4f && raw_depth > 0.0f) {
            ttc = raw_depth / speed;
            if (ttc > 50.0f) {
                ttc = -1.0f;  // 相对变化率 < 2%/s，无实际碰撞风险
            }
        }
        ttc_danger = updateTtcDanger(track_id, ttc, filtered.is_large_jump);
    }
    return MotionStateInfoRecord(direction, accel_state, filtered.velocity, ttc, ttc_danger);
}

MotionStateEngine::FilteredState MotionStateEngine::computeStateImpl(int    track_id,
                                                                     float  raw_value,
                                                                     double timestamp) {
    if (raw_value <= 0.0f) {
        FilteredState fs;
        fs.valid = false;
        return fs;
    }

    auto & state = kf_states_[track_id];

    // 卡尔曼滤波初始化：状态维度=3 [位置,速度,加速度]，测量维度=1（仅观测位置）
    if (!state.is_initialized) {
        // 状态转移矩阵 F 在预测时根据 dt 动态更新
        // x_k = x_{k-1} + v*dt + 0.5*a*dt^2, v_k = v_{k-1} + a*dt, a_k = a_{k-1}
        state.kf.init(3, 1, 0);

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

        state.kf.statePost   = (cv::Mat_<float>(3, 1) << raw_value, 0.0f, 0.0f);
        state.last_timestamp = timestamp;
        state.is_initialized = true;

        FilteredState fs;
        fs.position    = raw_value;
        fs.first_frame = true;
        return fs;
    }

    float dt = static_cast<float>(timestamp - state.last_timestamp);
    if (dt <= 0.0f) {
        dt = 0.033f;  // 兜底保护，假设默认30fps
    }

    state.kf.transitionMatrix.at<float>(0, 1) = dt;
    state.kf.transitionMatrix.at<float>(0, 2) = 0.5f * dt * dt;
    state.kf.transitionMatrix.at<float>(1, 2) = dt;

    state.kf.predict();

    const float predicted_value = state.kf.statePre.at<float>(0, 0);
    const bool  is_large_jump =
        predicted_value > 0.0f &&
        std::abs(raw_value - predicted_value) / predicted_value > bbox_jump_ratio_threshold_;

    // EMA 先平滑正常观测；异常跳变时保留预测值，避免检测框抖动污染速度
    if (!is_large_jump) {
        if (state.ema_value <= 0.0f) {
            state.ema_value = raw_value;
        } else {
            state.ema_value = ema_alpha_ * raw_value + (1.0f - ema_alpha_) * state.ema_value;
        }
    }
    const float filtered_value = is_large_jump ? predicted_value : state.ema_value;

    const cv::Mat measurement     = (cv::Mat_<float>(1, 1) << filtered_value);
    const cv::Mat estimated_state = state.kf.correct(measurement);

    state.last_timestamp = timestamp;

    FilteredState fs;
    fs.position      = estimated_state.at<float>(0, 0);
    fs.velocity      = estimated_state.at<float>(1, 0);
    fs.acceleration  = estimated_state.at<float>(2, 0);
    fs.is_large_jump = is_large_jump;
    return fs;
}

void MotionStateEngine::determineMotionStates(int           track_id,
                                              float         velocity,
                                              float         accel,
                                              MotionState & direction,
                                              MotionState & accel_state) {
    auto & state = kf_states_[track_id];

    // 迟滞规则：进入状态用高阈值（触发线），解除状态用低阈值（触发线 - 迟滞区间）。
    // 例如 velocity_threshold=20、velocity_hysteresis=5 时：速度 > 20 进入 APPROACH，
    // 已进入后要等速度回落到 < 15 才解除，避免在阈值边界来回抖动。
    direction   = state.prev_direction_state;
    accel_state = state.prev_accel_state;

    if (velocity > velocity_threshold_) {
        direction = MotionState::APPROACH;
    } else if (velocity < -velocity_threshold_) {
        direction = MotionState::MOVE_AWAY;
    } else {
        // 处于触发线以内：仅当回落越过解除线（触发线 - 迟滞）时才退出已激活的状态
        if (direction == MotionState::APPROACH &&
            velocity < velocity_threshold_ - velocity_hysteresis_) {
            direction = MotionState::STABLE;
        } else if (direction == MotionState::MOVE_AWAY &&
                   velocity > -velocity_threshold_ + velocity_hysteresis_) {
            direction = MotionState::STABLE;
        } else if (direction != MotionState::APPROACH && direction != MotionState::MOVE_AWAY) {
            direction = MotionState::STABLE;
        }
    }

    // 加速度迟滞：只在方向状态激活（趋近/远离）时判定加减速
    if (direction == MotionState::APPROACH || direction == MotionState::MOVE_AWAY) {
        if (accel > acceleration_threshold_) {
            accel_state = MotionState::ACCELE;
        } else if (accel < -acceleration_threshold_) {
            accel_state = MotionState::DECELE;
        } else {
            if (accel_state == MotionState::ACCELE &&
                accel < acceleration_threshold_ - acceleration_hysteresis_) {
                accel_state = MotionState::CONSTANT;
            } else if (accel_state == MotionState::DECELE &&
                       accel > -acceleration_threshold_ + acceleration_hysteresis_) {
                accel_state = MotionState::CONSTANT;
            } else if (accel_state != MotionState::ACCELE && accel_state != MotionState::DECELE) {
                accel_state = MotionState::CONSTANT;
            }
        }
    } else {
        // 方向状态未激活时，加速度状态复位，等待下一轮重新触发
        accel_state = MotionState::CONSTANT;
    }

    state.prev_direction_state = direction;
    state.prev_accel_state     = accel_state;
}

bool MotionStateEngine::updateTtcDanger(int track_id, float ttc, bool is_jump) {
    auto & state = kf_states_[track_id];

    // 跳变帧：尺度/速度不可信，ttc 无效。保持当前报警状态与计数不动，
    // 不当作"安全"帧去清零进入计数——否则抖动会让报警永远攒不满连续低帧
    if (is_jump) {
        return state.ttc_danger;
    }

    const bool low = ttc > 0.0f && ttc < ttc_warn_threshold_;
    const bool high = ttc <= 0.0f || ttc > ttc_clear_threshold_;  // 无效/静止/远离 或 已远离危险区

    if (low) {
        state.ttc_low_frames++;
        state.ttc_high_frames = 0;
    } else if (high) {
        state.ttc_low_frames = 0;
        state.ttc_high_frames++;
    }
    // [warn, clear] 灰色区：保持现状，防止在阈值边界抖动

    if (state.ttc_danger) {
        // 已报警：需连续 ttc_exit_frames 帧处于 high（高于 clear 或无效）才解除（阻塞保持）
        if (state.ttc_high_frames >= ttc_exit_frames_) {
            state.ttc_danger      = false;
            state.ttc_high_frames = 0;
            APP_INFO("[ControlPanel] track {} TTC alarm cleared: ttc={:.2f}s > clear {:.2f}s",
                     track_id, ttc, ttc_clear_threshold_);
        }
    } else if (state.ttc_low_frames >= ttc_enter_frames_) {
        // 未报警：需连续 ttc_enter_frames 帧低于 warn 才触发（进入防抖）
        state.ttc_danger      = true;
        state.ttc_low_frames  = 0;
        state.ttc_high_frames = 0;
        APP_INFO("[ControlPanel] track {} TTC alarm triggered: ttc={:.2f}s < warn {:.2f}s",
                 track_id, ttc, ttc_warn_threshold_);
    }
    return state.ttc_danger;
}

MotionStateInfoRecord MotionStateEngine::computeMotionStateFromBBox(const STrack & track,
                                                                    double         timestamp) {
    const auto & tlwh = track.tlwh_;
    if (tlwh.size() < 4 || tlwh[2] <= 0.0f || tlwh[3] <= 0.0f) {
        return MotionStateInfoRecord(MotionState::INVAILD, MotionState::INVAILD, 0.0f, -1.0f);
    }

    // 用 bbox 线性尺度代替深度：sqrt(area)
    const float scale    = std::sqrt(tlwh[2] * tlwh[3]);
    const auto  filtered = computeStateImpl(track.track_id_, scale, timestamp);

    MotionState direction   = MotionState::STABLE;
    MotionState accel_state = MotionState::CONSTANT;
    float       ttc         = -1.0f;
    bool        ttc_danger  = false;

    if (filtered.valid && !filtered.first_frame) {
        determineMotionStates(track.track_id_, filtered.velocity, filtered.acceleration, direction,
                              accel_state);

        // 跳变帧（is_large_jump）尺度突变，速度/尺度不可信，拒绝 TTC；
        // 速度过小（分母小 -> TTC 巨大）或非正、目标尺度低于下限同样视为无有效 TTC
        if (!filtered.is_large_jump && filtered.velocity > min_velocity_for_ttc_ &&
            scale > min_scale_for_ttc_) {
            ttc = scale / filtered.velocity;
        }

        ttc_danger = updateTtcDanger(track.track_id_, ttc, filtered.is_large_jump);
    }

    return MotionStateInfoRecord(direction, accel_state, filtered.velocity, ttc, ttc_danger);
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

            if (depth_value > 0.0f) {
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
