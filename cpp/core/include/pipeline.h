#pragma once
#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "motion_state_engine.h"
#include "yolo_detect_model.h"

class Pipeline {
  public:
    Pipeline(ConfigManager & config_manager, FrameMeta frame_meta);
    void init();

    // 核心推理接口，供正常业务和 Benchmark 调用
    void process(FrameInputContext &  frame_input_context,
                 InferOutputContext & infer_output_context);
    void processOverlap(FrameInputContext &  frame_input_context,
                        InferOutputContext & infer_output_context);

    cv::Scalar getColor(int idx) { return tracker_.getColor(idx); }

    YoloDetectModel & getDetector() { return detector_; }

    DepthModel & getDepthModel() { return depth_model_; }

    void postProcess(FrameInputContext &  frame_input_context,
                     InferOutputContext & infer_output_context);


  private:
    bool isTrackingClass(int class_id) {
        for (auto & c : track_classes_) {
            if (class_id == c) {
                return true;
            }
        }
        return false;
    }

    const ConfigManager & config_manager_;
    YoloDetectModel       detector_;
    DepthModel            depth_model_;
    BYTETracker           tracker_;
    MotionStateEngine     motion_state_engine_;

    // 跨帧缓存状态
    bool             has_cached_depth_ = false;
    cv::Mat          cached_depth_;
    cv::Mat          cached_depth_vis_;
    // 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
    std::vector<int> track_classes_{ 1, 2, 3, 5,
                                     7 };  // person, bicycle, car, motorcycle, bus, truck

    bool is_normalize_ = false;
};
