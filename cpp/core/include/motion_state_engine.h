#pragma once
#include "frame.h"
#include "STrack.h"

#include <cmath>
#include <cstdio>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
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

class MotionStateEngine {
  public:
    MotionStateEngine(float velocity_threshold       = 5.0f,
                      float acceleration_threshold   = 1.5f,
                      float kf_process_noise_cov     = 2e-2f,
                      float kf_measurement_noise_cov = 5e-2f);

    MotionStateInfoRecord computeMotionState(int track_id, float raw_depth, double timestamp);

    float getObjectDepth(cv::Mat depth, const STrack & track, cv::Size image_size);

    float computeMeanDepth(cv::Mat                    depth,
                           const std::vector<float> & tlwh,
                           int                        num_samples = 64) const;

  private:
    struct KalmanState {
        cv::KalmanFilter kf;
        double           last_timestamp;
        bool             is_initialized;
    };

    std::unordered_map<int, KalmanState> kf_states_;

    float velocity_threshold_;
    float acceleration_threshold_;
    float kf_process_noise_cov_;
    float kf_measurement_noise_cov_;
};
