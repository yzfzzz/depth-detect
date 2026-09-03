#include "pipeline.h"

#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "logger_manager.h"
#include "motion_state_engine.h"
#include "public.h"
#include "STrack.h"

#include <array>

Pipeline::Pipeline(ConfigManager & config_manager, FrameMeta frame_meta) :
    depth_enabled_(config_manager.isDepthEnabled()),
    tracker_(30, 30),  // 假设fps=30，或从config读取
    motion_state_engine_(config_manager.getMotionVelocityThreshold(),
                         config_manager.getMotionAccelerationThreshold(),
                         config_manager.getMotionVelocityHysteresis(),
                         config_manager.getMotionAccelerationHysteresis(),
                         config_manager.getKfProcessNoiseCov(),
                         config_manager.getKfMeasurementNoiseCov(),
                         config_manager.getMinScaleForTtc(),
                         config_manager.getMinVelocityForTtc(),
                         config_manager.getEmaAlpha(),
                         config_manager.getBboxJumpRatioThreshold(),
                         config_manager.getTtcWarnThreshold(),
                         config_manager.getTtcClearThreshold(),
                         config_manager.getTtcEnterFrames(),
                         config_manager.getTtcExitFrames()) {
    bool is_normalize = false;

    // 是否加载深度模型由 config 的 depth.enabled 控制
    if (config_manager.isDepthEnabled()) {
        depth_model_.init(config_manager.getDepthModelPath(), frame_meta.img_w, frame_meta.img_h,
                          is_normalize, config_manager.isUseGPU());
    }
    detector_.init(config_manager.getYoloModelPath(), frame_meta.img_w, frame_meta.img_h,
                   config_manager.getYoloNmsThresh(), config_manager.getYoloConfThresh(), 80,
                   config_manager.isUseGPU());

    // 是否记录每个 track 的类别/原始深度/面积/帧数等到 CSV
    if (config_manager.isTrackLogEnabled()) {
        track_log_enabled_ = true;
        track_log_path_    = config_manager.getOutDir() + "/track_log.csv";
        APP_INFO("Track log enabled: {}", track_log_path_);
    }
}

Pipeline::Pipeline(std::string depth_model_path,
                   std::string yolo_model_path,
                   FrameMeta   frame_meta,
                   bool        use_gpu,
                   float       yolo_nms_thresh,
                   float       yolo_conf_thresh) {
    bool        is_normalize = false;
    std::string backend_type = use_gpu ? "engine" : "onnx";
    depth_enabled_           = true;
    depth_model_.init(
        {
            { backend_type, depth_model_path }
    },
        frame_meta.img_w, frame_meta.img_h, is_normalize, use_gpu);
    detector_.init(
        {
            { backend_type, yolo_model_path }
    },
        frame_meta.img_w, frame_meta.img_h, yolo_nms_thresh, yolo_conf_thresh, 80, use_gpu);
}

Pipeline::~Pipeline() {
    if (track_log_enabled_ && !track_log_data_.empty()) {
        LoggerManager::saveTrackCsv(track_log_path_, track_log_data_);
    }
}

void Pipeline::init() {}

void Pipeline::process(FrameInputContext &  frame_input_context,
                       InferOutputContext & infer_output_context) {
    detector_.runInference(frame_input_context, infer_output_context);
    if (depth_enabled_) {
        depth_model_.runInference(frame_input_context, infer_output_context);
    }
    updateTracker(infer_output_context);
    updateMotionStates(frame_input_context, infer_output_context);
}

void Pipeline::processOverlap(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context) {
    detector_.runInferenceAsync(frame_input_context);
    if (depth_enabled_) {
        depth_model_.runInferenceAsync(frame_input_context);
    }
    detector_.getInferOutputResult(infer_output_context);
    updateTracker(infer_output_context);
    if (depth_enabled_) {
        depth_model_.getInferOutputResult(infer_output_context);
    }
    updateMotionStates(frame_input_context, infer_output_context);
}

void Pipeline::updateMotionStates(FrameInputContext &  frame_input_context,
                                  InferOutputContext & infer_output_context) {
    infer_output_context.motion_records.clear();

    // 深度推理开启且本帧有原始视差输出时，用深度做运动状态/TTC 估计，否则退回 bbox 尺度
    cv::Mat depth_metric;
    if (depth_enabled_ && !infer_output_context.depth_raw_infer_out.empty()) {
        const auto dims = depth_model_.getInputDims();
        if (dims.size() >= 4 && dims[2] > 0 && dims[3] > 0) {
            cv::Mat raw_depth_map(dims[2], dims[3], CV_32FC1,
                                  infer_output_context.depth_raw_infer_out.data());
            cv::resize(raw_depth_map, depth_metric,
                       cv::Size(frame_input_context.meta.img_w, frame_input_context.meta.img_h));
        }
    }

    for (const auto & track : infer_output_context.tracked_objects) {
        if (track.tlwh_[2] * track.tlwh_[3] <= 20) {
            continue;
        }
        float raw_depth = 0.0f;
        if (!depth_metric.empty()) {
            raw_depth = motion_state_engine_.computeMeanDepth(depth_metric, track.tlwh_);
        }
        // 深度与 bbox 两路都执行，各自独立滤波状态（depth_kf_states_ / bbox_kf_states_），互不污染
        const MotionStateInfoRecord depth_motion = motion_state_engine_.computeMotionStateFromDepth(
            track.track_id_, raw_depth, frame_input_context.timestamp);
        const MotionStateInfoRecord bbox_motion =
            motion_state_engine_.computeMotionStateFromBBox(track, frame_input_context.timestamp);

        // 下游使用：深度有效用深度，否则退回 bbox
        const MotionStateInfoRecord & motion = bbox_motion;
        // const MotionStateInfoRecord & motion = raw_depth > 0.0f ? depth_motion : bbox_motion;
        infer_output_context.motion_records.insert({ track.track_id_, motion });

        if (track_log_enabled_) {
            track_log_data_[track.track_id_].push_back(
                { frame_input_context.frame_id, track.class_id_, track.tlwh_[2] * track.tlwh_[3],
                  raw_depth, depth_motion.velocity, bbox_motion.velocity, depth_motion.ttc,
                  depth_motion.ttc_danger, bbox_motion.ttc, bbox_motion.ttc_danger });
        }
    }
}

void Pipeline::updateTracker(InferOutputContext & infer_output_context) {
    // 从检测结果中筛选需要跟踪的类别（person/bicycle/car/motorcycle/bus/truck）
    std::vector<Detection> & res = infer_output_context.detections;
    std::vector<Object>      objects;
    for (size_t j = 0; j < res.size(); j++) {
        if (isTrackingClass(res[j].classId)) {
            cv::Rect_<float> rect(res[j].bbox[0], res[j].bbox[1], (res[j].bbox[2] - res[j].bbox[0]),
                                  (res[j].bbox[3] - res[j].bbox[1]));
            objects.push_back({ rect, res[j].classId, res[j].conf });
        }
    }
    infer_output_context.tracked_objects = tracker_.update(objects);
}
