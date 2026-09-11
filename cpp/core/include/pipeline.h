#pragma once
#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "logger_manager.h"
#include "motion_state_engine.h"
#include "yolo_depth_model.h"
#include "yolo_detect_model.h"

class Pipeline {
  public:
    Pipeline(ConfigManager & config_manager, FrameMeta frame_meta);
    Pipeline(std::string depth_model_path,
             std::string yolo_model_path,
             FrameMeta   frame_meta,
             bool        use_gpu          = false,
             float       yolo_nms_thresh  = 0.4f,
             float       yolo_conf_thresh = 0.25f);
    ~Pipeline();
    void init();

    // 核心推理接口，供正常业务和 Benchmark 调用
    void process(FrameInputContext &  frame_input_context,
                 InferOutputContext & infer_output_context);
    void processOverlap(FrameInputContext &  frame_input_context,
                        InferOutputContext & infer_output_context);

    void updateMotionStates(FrameInputContext &  frame_input_context,
                            InferOutputContext & infer_output_context);

    cv::Scalar getColor(int idx) { return tracker_.getColor(idx); }

    YoloDetectModel & getDetector() { return detector_; }

    LiteMonoDepthModel & getDepthModel() { return depth_model_; }

    BYTETracker & getTracker() { return tracker_; }

    MotionStateEngine & getMotionStateEngine() { return motion_state_engine_; }

  private:
    void updateTracker(InferOutputContext & infer_output_context);

    YoloDetectModel    detector_;
    LiteMonoDepthModel depth_model_;
    YoloDepthModel     yolo_depth_model_;
    bool               depth_enabled_ = false;  // 由 config 的 depth.enabled 控制

    bool isTrackingClass(int class_id) {
        for (auto & c : track_classes_) {
            if (class_id == c) {
                return true;
            }
        }
        return false;
    }

    // 参与快速靠近（approach）判定的类别：bicycle/car/motorcycle/bus/truck
    // person 只跟踪、不参与接近判定（人车各自独立，不合并）
    bool isApproachClass(int class_id) {
        for (auto & c : approach_classes_) {
            if (class_id == c) {
                return true;
            }
        }
        return false;
    }

    BYTETracker       tracker_;
    MotionStateEngine motion_state_engine_;

    // 跨帧缓存状态
    bool                                              has_cached_depth_ = false;
    cv::Mat                                           cached_depth_;
    cv::Mat                                           cached_depth_vis_;
    // 累积每个 track 的逐帧记录，运行结束时由 LoggerManager 统一写入 CSV
    bool                                              track_log_enabled_ = false;
    std::string                                       track_log_path_;
    std::unordered_map<int, std::vector<TrackRecord>> track_log_data_;
    // 需要跟踪的类别（对应 COCO 数据集类别索引）：person 也跟踪，但不做接近判定
    std::vector<int>                                  track_classes_{
        0, 1, 2, 3, 5, 7
    };  // person, bicycle, car, motorcycle, bus, truck
    // 参与接近判定的类别：bicycle, car, motorcycle, bus, truck
    std::vector<int> approach_classes_{ 1, 2, 3, 5, 7 };

    bool is_normalize_ = false;
};
