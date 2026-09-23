#pragma once
#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "logger_manager.h"
#include "motion_state_engine.h"
#include "yolo_depth_model.h"
#include "yolo_detect_model.h"

// 调度策略
enum class ScheduleMode {
    SYNC,     // 串行：检测跑完再跑深度（overlap=false，或后端不支持异步时的回落）
    OVERLAP,  // 重叠：检测与深度同帧、各自异步提交到独立 stream 重叠（overlap=true 且 TRT 后端）
    STAGGER,  // 错峰：检测/深度按各自间隔独立调度，互补帧只跑一路，同帧碰撞时复用重叠逻辑
};

// 供日志输出使用的策略名
const char * scheduleModeName(ScheduleMode mode);

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

    // 唯一推理入口：内部按 schedule_mode_ 分发到下面三个分支，业务侧只调这一个。
    void process(FrameInputContext &  frame_input_context,
                 InferOutputContext & infer_output_context);

    // 串行：检测 → 跟踪 → 深度 → 运动状态，GPU 上不会出现两模型并发
    void processSync(FrameInputContext &  frame_input_context,
                     InferOutputContext & infer_output_context);
    // 重叠：本帧检测与深度都跑，异步并行提交；非 TensorRT 后端自动降级为同步
    void processOverlap(FrameInputContext &  frame_input_context,
                        InferOutputContext & infer_output_context);
    // 错峰：按各自间隔调度检测/深度，互补帧只跑一路（另一路沿用上帧结果），
    // 两路撞到同一帧时转发 processOverlap
    void processStagger(FrameInputContext &  frame_input_context,
                        InferOutputContext & infer_output_context);

    ScheduleMode getScheduleMode() const { return schedule_mode_; }
    void         setScheduleMode(ScheduleMode mode);

    void updateMotionStates(FrameInputContext &  frame_input_context,
                            InferOutputContext & infer_output_context);

    cv::Scalar getColor(int idx) { return tracker_.getColor(idx); }

    YoloDetectModel & getDetector() { return detector_; }

    BaseModel & getActiveDepthModel() {
        return use_yolo_depth_ ? static_cast<BaseModel &>(yolo_depth_model_) :
                                 static_cast<BaseModel &>(depth_model_);
    }

    BYTETracker & getTracker() { return tracker_; }

    MotionStateEngine & getMotionStateEngine() { return motion_state_engine_; }

  private:
    void updateTracker(InferOutputContext & infer_output_context);
    void runDepthInference(FrameInputContext &  frame_input_context,
                           InferOutputContext & infer_output_context);

    // 构造期解析调度策略写入 schedule_mode_
    void resolveScheduleMode(bool overlap_requested);

    // 错峰下的间隔调度：frame_id % interval == 0 的帧才触发对应模型推理。
    // 只在 STAGGER 策略下生效，串行/重叠策略每帧两路都跑
    bool shouldRunDetect(const FrameInputContext & frame_input_context) const {
        return schedule_mode_ != ScheduleMode::STAGGER ||
               (frame_input_context.frame_id % detect_interval_ == 0);
    }
    bool shouldRunDepth(const FrameInputContext & frame_input_context) const {
        if (!depth_enabled_) {
            return false;
        }
        return schedule_mode_ != ScheduleMode::STAGGER ||
               (frame_input_context.frame_id % depth_interval_ == 0);
    }

    // 串行分支共用的执行与收尾：processSync（两路都跑）与 processStagger 的单路帧共用。
    // 收尾顺序与 processOverlap 保持一致：检测 → 跟踪 → 深度 → 运动状态；
    // 跟踪放在深度前是为了让 CPU 侧的跟踪计算能与深度推理的等待重叠
    void runBranchesSerially(FrameInputContext &  frame_input_context,
                             InferOutputContext & infer_output_context,
                             bool                 run_detect,
                             bool                 run_depth);

    // TensorRT 后端支持异步提交（双 stream 重叠）；ONNX Runtime 无异步接口，
    static bool isAsyncCapable(const BaseModel & model) {
        return model.getBackendType() == BackendType::TensorRT;
    }

    // 重叠帧使用的检测模型：优先轻量模型，未加载则回落主模型。
    YoloDetectModel & overlapDetector() {
        return (has_light_detector_ && use_yolo_depth_) ? detector_light_ : detector_;
    }

    YoloDetectModel detector_;        // 主检测模型（yaml 配置，如 yolo26s）
    YoloDetectModel detector_light_;  // 碰撞帧专用轻量模型（yolo26n），仅错峰时加载
    bool               has_light_detector_ = false;  // 轻量模型是否可用
    LiteMonoDepthModel depth_model_;
    YoloDepthModel     yolo_depth_model_;
    bool               depth_enabled_   = false;  // 由 config 的 depth.enabled 控制
    bool               stagger_infer_   = false;  // 错峰推理总开关
    int                detect_interval_ = 1;      // 检测推理间隔：N = 每 N 帧推 1 次
    int                depth_interval_  = 1;      // 深度推理间隔：N = 每 N 帧推 1 次
    bool use_yolo_depth_ = true;  // 深度模型类型：true = yolo_depth，false = lite_mono
    // 本次运行的调度策略，构造期由 resolveScheduleMode 按配置解析（默认串行）
    ScheduleMode schedule_mode_ = ScheduleMode::SYNC;

    bool isTrackingClass(int class_id) {
        for (auto & c : track_classes_) {
            if (class_id == c) {
                return true;
            }
        }
        return false;
    }

    BYTETracker       tracker_;
    MotionStateEngine motion_state_engine_;

    // 累积每个 track 的逐帧记录，运行结束时由 LoggerManager 统一写入 CSV
    bool                                              track_log_enabled_ = false;
    std::string                                       track_log_path_;
    std::unordered_map<int, std::vector<TrackRecord>> track_log_data_;
    std::vector<int>                                  track_classes_{
        COCO80::BICYCLE, COCO80::CAR, COCO80::MOTORCYCLE, COCO80::BUS, COCO80::TRUCK
    };  // bicycle, car, motorcycle, bus, train, truck
    bool is_normalize_ = false;
};
