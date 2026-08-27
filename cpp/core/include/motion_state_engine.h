#pragma once
#include "frame.h"
#include "STrack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

const std::map<std::pair<MotionState, MotionState>, std::string> MOTION_STR_MAP = {
    { { MotionState::STABLE, MotionState::CONSTANT },    "Stable"               },
    { { MotionState::APPROACH, MotionState::ACCELE },    "Approach (Accele)"    },
    { { MotionState::APPROACH, MotionState::DECELE },    "Approach (Decele)"    },
    { { MotionState::APPROACH, MotionState::CONSTANT },  "Approach (Constant)"  },
    { { MotionState::MOVE_AWAY, MotionState::ACCELE },   "Move Away (Accele)"   },
    { { MotionState::MOVE_AWAY, MotionState::DECELE },   "Move Away (Decele)"   },
    { { MotionState::MOVE_AWAY, MotionState::CONSTANT }, "Move Away (Constant)" },
};

// 运动状态估计引擎：滤波与运动状态/TTC/报警判定分层。
// 滤波层（computeStateImpl）只输出位置/速度/加速度与跳变标志，不做状态判定；
// computeMotionStateFromBBox 依据 bbox 尺度做状态判定并计算 TTC——跳变帧与过小的
// 尺度变化速度不产生有效 TTC（避免尺度突变污染、分母过小得到巨大假值），
// TTC 危险报警按连续低/高帧数判定（延迟阻塞），防止阈值边界抖动。
class MotionStateEngine {
  public:
    MotionStateEngine(float velocity_threshold     = 5.0f,
                      float acceleration_threshold = 1.5f,
                      float velocity_hysteresis = 2.0f,  // 速度迟滞区间（解除线 = 触发线 - 迟滞）
                      float acceleration_hysteresis  = 1.0f,  // 加速度迟滞区间
                      float kf_process_noise_cov     = 2e-2f,
                      float kf_measurement_noise_cov = 5e-2f,
                      float min_scale_for_ttc = 20.0f,  // bbox 线性尺度下限（sqrt(w*h)）
                      float min_velocity_for_ttc = 1.0f,  // 尺度变化速度下限：分母过小则 TTC 无意义
                      float ema_alpha                 = 0.3f,
                      float bbox_jump_ratio_threshold = 0.35f,  // 尺度变化异常阈值
                      float ttc_warn_threshold        = 3.0f,   // TTC 报警触发阈值（秒）
                      float ttc_clear_threshold = 4.0f,  // TTC 报警解除阈值（秒，须 >= warn）
                      int ttc_enter_frames = 3,  // 连续低于 warn 帧数后触发报警（防抖）
                      int ttc_exit_frames = 10);  // 连续高于 clear/无效帧数后才解除（阻塞）
    // 基于深度/视差计算运动状态、TTC 与 TTC 危险报警（TTC = 深度值/趋近速度）
    MotionStateInfoRecord computeMotionStateFromDepth(int    track_id,
                                                      float  raw_depth,
                                                      double timestamp);
    // 基于目标框尺度变化计算运动状态、TTC 与 TTC 危险报警
    MotionStateInfoRecord computeMotionStateFromBBox(const STrack & track, double timestamp);

    float getObjectDepth(cv::Mat depth, const STrack & track, cv::Size image_size);

    float computeMeanDepth(cv::Mat                    depth,
                           const std::vector<float> & tlwh,
                           int                        num_samples = 64) const;

  private:
    // 纯滤波输出：只含滤波状态，不含运动状态判定
    struct FilteredState {
        float position     = 0.0f;  // 滤波后的位置（尺度/深度）
        float velocity     = 0.0f;  // 卡尔曼估计速度
        float acceleration = 0.0f;  // 卡尔曼估计加速度
        bool is_large_jump = false;  // 本帧判为大跳变（速度/尺度不可信，TTC 应拒绝）
        bool first_frame = false;  // 滤波器初始化帧（无历史，不做状态/TTC 判定）
        bool valid       = true;   // 输入非法（<=0）时为 false
    };

    struct KalmanState {
        cv::KalmanFilter kf;
        double           last_timestamp;
        bool             is_initialized;
        float            ema_value            = 0.0f;
        // 迟滞状态机需要记住上一帧的方向/加速度状态，才能实现"进入用触发线、解除用触发线-迟滞"
        MotionState      prev_direction_state = MotionState::STABLE;
        MotionState      prev_accel_state     = MotionState::CONSTANT;
        // TTC 危险报警的延迟阻塞状态
        int              ttc_low_frames       = 0;  // 连续 ttc < warn 的帧数（进入防抖）
        int  ttc_high_frames = 0;  // 连续 ttc > clear 或无效的帧数（退出保持）
        bool ttc_danger      = false;
    };

    // 公共卡尔曼滤波逻辑：位置/速度/加速度（纯滤波，不判定运动状态）
    FilteredState computeStateImpl(int track_id, float raw_value, double timestamp);

    // 运动状态判定（迟滞防抖）：读写 kf_states_ 内持久化的上一帧状态
    void determineMotionStates(int           track_id,
                               float         velocity,
                               float         accel,
                               MotionState & direction,
                               MotionState & accel_state);

    // TTC 危险报警（延迟阻塞）：连续低于 warn 帧数后触发，
    // 连续高于 clear 或无效帧数后才解除；跳变帧（is_jump）中性处理，不参与计数
    bool updateTtcDanger(int track_id, float ttc, bool is_jump);

    std::unordered_map<int, KalmanState> kf_states_;

    float velocity_threshold_;
    float acceleration_threshold_;
    float velocity_hysteresis_;
    float acceleration_hysteresis_;
    float kf_process_noise_cov_;
    float kf_measurement_noise_cov_;
    float min_scale_for_ttc_;
    float min_velocity_for_ttc_;
    float ema_alpha_;
    float bbox_jump_ratio_threshold_;
    float ttc_warn_threshold_;
    float ttc_clear_threshold_;
    int   ttc_enter_frames_;
    int   ttc_exit_frames_;

  public:
    void setVelocityThreshold(float value) { velocity_threshold_ = std::max(0.0f, value); }

    void setAccelerationThreshold(float value) { acceleration_threshold_ = std::max(0.0f, value); }

    void setVelocityHysteresis(float value) { velocity_hysteresis_ = std::max(0.0f, value); }

    void setAccelerationHysteresis(float value) {
        acceleration_hysteresis_ = std::max(0.0f, value);
    }

    float getVelocityThreshold() const { return velocity_threshold_; }

    float getAccelerationThreshold() const { return acceleration_threshold_; }

    float getVelocityHysteresis() const { return velocity_hysteresis_; }

    float getAccelerationHysteresis() const { return acceleration_hysteresis_; }

    float getEmaAlpha() const { return ema_alpha_; }

    float getBboxJumpRatioThreshold() const { return bbox_jump_ratio_threshold_; }

    void setEmaAlpha(float value) { ema_alpha_ = std::min(1.0f, std::max(0.0f, value)); }

    void setBboxJumpRatioThreshold(float value) {
        bbox_jump_ratio_threshold_ = std::max(0.0f, value);
    }

    void setTtcWarnThreshold(float value) {
        ttc_warn_threshold_ = std::max(0.0f, value);
        // warn 上调时联动抬高 clear，避免 warn > clear 导致报警永不解除
        if (ttc_clear_threshold_ < ttc_warn_threshold_) {
            ttc_clear_threshold_ = ttc_warn_threshold_;
        }
    }

    void setTtcClearThreshold(float value) {
        ttc_clear_threshold_ = std::max(ttc_warn_threshold_, value);
    }

    float getTtcWarnThreshold() const { return ttc_warn_threshold_; }

    float getTtcClearThreshold() const { return ttc_clear_threshold_; }

    void setTtcEnterFrames(int value) { ttc_enter_frames_ = std::max(1, value); }

    void setTtcExitFrames(int value) { ttc_exit_frames_ = std::max(1, value); }

    int getTtcEnterFrames() const { return ttc_enter_frames_; }

    int getTtcExitFrames() const { return ttc_exit_frames_; }
};
