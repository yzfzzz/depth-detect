#include "pipeline.h"

#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "logger_manager.h"
#include "motion_state_engine.h"
#include "public.h"
#include "STrack.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>

double durationMs(const std::chrono::steady_clock::time_point & begin,
                  const std::chrono::steady_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool fileReadable(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::ifstream file(path);
    return file.good();
}

Pipeline::Pipeline(ConfigManager & config_manager, FrameMeta frame_meta) :
    depth_enabled_(config_manager.isDepthEnabled()),
    stagger_infer_(config_manager.isStaggerInferEnabled()),
    detect_interval_(std::max(1, config_manager.getYoloDetectInterval())),
    depth_interval_(std::max(1, config_manager.getDepthInterval())),
    tracker_(static_cast<int>(60, 0.3, 0.1, 0.5, 0.8)) {
    bool is_normalize = false;

    APP_INFO(
        "ByteTracker params: max_time_lost={}, track_high_thresh={}, track_low_thresh={}, "
        "new_track_thresh={}, match_thresh={}",
        static_cast<int>(config_manager.getCameraFps() / 30.0 *
                         config_manager.getTrackerTrackBuffer()),
        config_manager.getTrackHighThresh(), config_manager.getTrackLowThresh(),
        config_manager.getNewTrackThresh(), config_manager.getMatchThresh());

    // 快速靠近检测参数
    approach::ApproachParams approach_params;
    approach_params.detect_warmup   = config_manager.getApproachDetectWarmup();
    approach_params.depth_warmup    = config_manager.getApproachDepthWarmup();
    approach_params.thr_depth       = config_manager.getApproachThrDepth();
    approach_params.thr_height      = config_manager.getApproachThrHeight();
    approach_params.detect_recent_w = config_manager.getApproachDetectRecentW();
    approach_params.depth_recent_w  = config_manager.getApproachDepthRecentW();
    approach_params.score_thr       = config_manager.getApproachScoreThr();
    approach_params.confirm         = config_manager.getApproachConfirm();
    approach_params.exit_score_thr  = config_manager.getApproachExitScoreThr();
    approach_params.exit_confirm    = config_manager.getApproachExitConfirm();

    // 未识别的取值回落到 none 并告警
    const std::string    filter_name = config_manager.getApproachFilterMode();
    approach::FilterMode filter_mode = approach::parseFilterMode(filter_name);
    if (filter_mode == approach::FilterMode::NONE && filter_name != "none") {
        APP_WARN("Unknown approach filter '{}', fallback to none", filter_name);
    }
    motion_state_engine_.configureApproach(approach_params, filter_mode,
                                           config_manager.isApproachEnabled());

    // 是否加载深度模型由 config 的 depth.enabled 控制
    use_yolo_depth_ = (config_manager.getDepthModelType() == "yolo_depth");
    if (config_manager.isDepthEnabled()) {
        if (config_manager.getDepthModelType() == "lite_mono") {
            depth_model_.init(config_manager.getDepthModelPath(), frame_meta.img_w,
                              frame_meta.img_h, is_normalize, config_manager.isUseGPU());
        } else if (use_yolo_depth_) {
            yolo_depth_model_.init(config_manager.getDepthModelPath(), frame_meta.img_w,
                                   frame_meta.img_h, config_manager.isUseGPU());
        } else {
            APP_ERROR("Unsupported depth model type: {}", config_manager.getDepthModelType());
        }
    }
    // 主检测模型, 非碰撞帧用它保精度
    std::map<std::string, std::string> yolo_model_paths = config_manager.getYoloModelPath();
    detector_.init(yolo_model_paths, frame_meta.img_w, frame_meta.img_h,
                   config_manager.getYoloNmsThresh(), config_manager.getYoloConfThresh(), 80,
                   config_manager.isUseGPU());

    // 错峰调度下碰撞帧（检测与深度同帧）走重叠推理，主模型与深度模型同帧抢占 GPU
    // 延迟过大 → 加载 yaml 配置的轻量模型（yolo26n）专用于碰撞帧；
    // 两实例各自持有独立的 engine/context，GPU 显存会多占一份。
    // 路径按"双臂"构建：engine 优先、onnx 兜底（CPU 模式或 GPU 不可用时 base_model
    // 会自动回落到 onnx），因此降级到 onnx 后端同样能正常出结果
    if (stagger_infer_ && depth_enabled_) {
        const std::string light_engine_path = config_manager.getYoloLightEnginePath();
        const std::string light_onnx_path   = config_manager.getYoloLightOnnxPath();

        std::map<std::string, std::string> light_paths;
        if (fileReadable(light_engine_path)) {
            light_paths["engine"] = light_engine_path;
        }
        if (fileReadable(light_onnx_path)) {
            light_paths["onnx"] = light_onnx_path;
        }

        if (!light_paths.empty()) {
            detector_light_.init(light_paths, frame_meta.img_w, frame_meta.img_h,
                                 config_manager.getYoloNmsThresh(),
                                 config_manager.getYoloConfThresh(), 80, config_manager.isUseGPU());
            has_light_detector_ = detector_light_.isBackendInitialized();
            if (has_light_detector_) {
                APP_INFO(
                    "[Pipeline] overlap-frame light detector: engine='{}', onnx='{}', "
                    "backend={}",
                    light_paths.count("engine") ? light_engine_path : std::string("<none>"),
                    light_paths.count("onnx") ? light_onnx_path : std::string("<none>"),
                    detector_light_.backendTypeName().c_str());
            } else {
                APP_WARN(
                    "[Pipeline] light detector init failed, overlap frames reuse main detector");
            }
        } else {
            APP_WARN(
                "[Pipeline] yolo.light_engine / light_onnx neither configured nor readable; "
                "overlap frames will reuse the main detector");
        }
    }

    if (stagger_infer_ && depth_enabled_) {
        APP_INFO(
            "[Pipeline] staggered inference enabled: detect every {} frame(s), depth every {} "
            "frame(s); frames where both fire use overlapped inference "
            "(skipped model reuses last result)",
            detect_interval_, depth_interval_);
    }

    // 是否记录每个 track 的类别/原始深度/面积/帧数等到 CSV
    if (config_manager.isTrackLogEnabled()) {
        track_log_enabled_ = true;
        track_log_path_    = "track_log.csv";
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

void Pipeline::runDepthInference(FrameInputContext &  frame_input_context,
                                 InferOutputContext & infer_output_context) {
    if (use_yolo_depth_) {
        yolo_depth_model_.runInference(frame_input_context, infer_output_context);
    } else {
        depth_model_.runInference(frame_input_context, infer_output_context);
    }
}

void Pipeline::process(FrameInputContext &  frame_input_context,
                       InferOutputContext & infer_output_context) {
    // 间隔调度（stagger_infer 开启时生效）：frame_id % interval == 0 的帧触发对应模型推理，
    // interval=1 每帧推、2 隔帧推、3 隔2帧推。被跳过的一路沿用上一帧结果：
    // 检测帧不更新深度图（depth_raw_infer_out 保持），深度帧不调用 updateTracker
    const bool stagger      = stagger_infer_ && depth_enabled_;
    const bool detect_frame = !stagger || (frame_input_context.frame_id % detect_interval_ == 0);
    const bool depth_frame =
        depth_enabled_ && (!stagger || (frame_input_context.frame_id % depth_interval_ == 0));

    // 两路调度到同一帧：走重叠推理（异步并行提交，算力重叠利用）。
    // 重叠实现仅 yolo_depth 支持（异步双 stream），lite_mono 回落为串行同帧执行
    if (stagger && detect_frame && depth_frame && use_yolo_depth_) {
        processOverlap(frame_input_context, infer_output_context);
        return;
    }

    const auto t_detect_begin = std::chrono::steady_clock::now();
    if (detect_frame) {
        detector_.runInference(frame_input_context, infer_output_context);
    }

    double depth_ms = 0.0;
    if (depth_frame) {
        const auto t_depth_begin = std::chrono::steady_clock::now();
        runDepthInference(frame_input_context, infer_output_context);
        const auto t_depth_end = std::chrono::steady_clock::now();
    }

    if (detect_frame) {
        updateTracker(infer_output_context);
    }
    updateMotionStates(frame_input_context, infer_output_context);
}

void Pipeline::processOverlap(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context) {
    // 重叠帧用轻量检测模型（yolo26n），降低与深度模型同帧抢占 GPU 的延迟
    YoloDetectModel & detector = overlapDetector();

    // 异步重叠只有 TensorRT 后端支持（双 stream 并行提交）；ONNX Runtime 没有异步接口
    // （BaseModel::runInferenceAsync 会直接报错返回 false），该路退回同步执行——
    // 降级到 onnx 后端时依然能正常出结果，只是失去重叠收益
    const bool detect_async = (detector.getBackendType() == BackendType::TensorRT);
    const bool depth_async  = depth_enabled_ && use_yolo_depth_ &&
                             (yolo_depth_model_.getBackendType() == BackendType::TensorRT);

    // 先提交深度异步（TRT），让深度在 GPU 上跑的同时 CPU 侧推进检测，最大化重叠窗口
    if (depth_async) {
        yolo_depth_model_.runInferenceAsync(frame_input_context);
    }

    // TRT 走异步提交，其他后端走同步
    if (detect_async) {
        detector.runInferenceAsync(frame_input_context);
    } else {
        detector.runInference(frame_input_context, infer_output_context);
    }

    if (detect_async) {
        detector.getInferOutputResult(infer_output_context);
    }
    updateTracker(infer_output_context);

    if (depth_enabled_) {
        if (depth_async) {
            yolo_depth_model_.getInferOutputResult(infer_output_context);
        } else {
            // 非异步深度（ONNX 后端 / lite_mono）：此处同步执行，由 runDepthInference
            // 按 yolo_depth / lite_mono 分发到实际初始化的那个模型
            runDepthInference(frame_input_context, infer_output_context);
        }
    }
    updateMotionStates(frame_input_context, infer_output_context);
}

void Pipeline::updateMotionStates(FrameInputContext &  frame_input_context,
                                  InferOutputContext & infer_output_context) {
    infer_output_context.motion_records.clear();

    // 深度推理开启且本帧有原始视差输出时，构造与图像同分辨率的深度图（供“框内取深度”用）；
    // 深度不可用时接近判定拿到 0 深度（视为无效，检测器不产生虚假变化）
    cv::Mat depth_metric;
    if (depth_enabled_ && !infer_output_context.depth_raw_infer_out.empty()) {
        // 输入尺寸取自实际执行推理的那个深度模型
        const auto dims =
            use_yolo_depth_ ? yolo_depth_model_.getInputDims() : depth_model_.getInputDims();
        if (dims.size() >= 4 && dims[2] > 0 && dims[3] > 0) {
            cv::Mat raw_depth_map(dims[2], dims[3], CV_32FC1,
                                  infer_output_context.depth_raw_infer_out.data());
            cv::resize(raw_depth_map, depth_metric,
                       cv::Size(frame_input_context.meta.img_w, frame_input_context.meta.img_h));
        }
    }

    // 快速靠近检测：逐 track 独立判定
    // 参与判定的类别只有 bicycle/car/motorcycle/bus/truck
    // 每个 track 用自己的框高 + 框内深度送进 updateApproachState，
    for (auto & track : infer_output_context.tracked_objects) {
        approach::ApproachState approach_state;
        bool                    has_approach = false;
        float                   depth        = 0.0f;
        if (track.tlwh[3] * track.tlwh[2] < 400) {
            continue;
        }

        if (isTrackingClass(track.class_id)) {
            // 框内鲁棒深度估计（深度不可用时为 0，检测器视为无效值）
            if (!depth_metric.empty()) {
                depth           = motion_state_engine_.computeMeanDepth(depth_metric, track.tlwh);
                track.distance_ = depth;
            }
            // 目标框高度 + 框内深度 -> 逐帧接近判定（track_id 即判定状态索引）
            approach_state = motion_state_engine_.updateApproachState(
                track.track_id, track.tlwh[3], depth, frame_input_context.timestamp);
            has_approach = true;
        }

        // 记录本帧判定结果
        MotionStateInfoRecord motion;
        if (has_approach) {
            motion.approach_alarm       = approach_state.alarm;
            motion.approach_score       = approach_state.score;
            motion.approach_depth_score = approach_state.depth_score;
            motion.approach_scale_score = approach_state.scale_score;
        }
        infer_output_context.motion_records.emplace(track.track_id, motion);

        // track_log：每个目标框一行
        if (track_log_enabled_) {
            float raw_depth = depth;
            if (!has_approach && !depth_metric.empty()) {
                raw_depth = motion_state_engine_.computeMeanDepth(depth_metric, track.tlwh);
            }
            track_log_data_[track.track_id].push_back(
                { frame_input_context.frame_id, track.class_id, track.tlwh[0], track.tlwh[2],
                  track.tlwh[1], track.tlwh[3], track.tlwh[2] * track.tlwh[3], raw_depth });
        }
    }
}

void Pipeline::updateTracker(InferOutputContext & infer_output_context) {
    // 从检测结果中筛选需要跟踪的类别（bicycle/car/motorcycle/bus/truck）
    std::vector<Detection> & res = infer_output_context.detections;
    std::vector<Object>      objects;
    for (size_t j = 0; j < res.size(); j++) {
        if (isTrackingClass(res[j].classId)) {
            cv::Rect_<float> rect(res[j].bbox[0], res[j].bbox[1], (res[j].bbox[2] - res[j].bbox[0]),
                                  (res[j].bbox[3] - res[j].bbox[1]));
            objects.push_back({ rect, res[j].classId, res[j].conf, 0.0f });
        }
    }
    infer_output_context.tracked_objects.clear();
    std::vector<STrack> lost_stracks;  // 本帧丢失轨迹，暂时不使用，兼容接口
    tracker_.update(objects, lost_stracks, infer_output_context.tracked_objects);
}
