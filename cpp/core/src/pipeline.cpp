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
#include <chrono>

namespace {
double durationMs(const std::chrono::steady_clock::time_point & begin,
                  const std::chrono::steady_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
}  // namespace

Pipeline::Pipeline(ConfigManager & config_manager, FrameMeta frame_meta) :
    depth_enabled_(config_manager.isDepthEnabled()),
    tracker_(30, 30) {
    bool is_normalize = false;

    // 快速靠近（approach）检测：参数取自 config.motion_state_engine.approach，
    // 默认值即 mini_python/pipeline.py 调好的最优参数（方案 d + 1€ 滤波）
    approach::ApproachParams approach_params;
    approach_params.warmup         = config_manager.getApproachWarmup();
    approach_params.thr_depth      = config_manager.getApproachThrDepth();
    approach_params.thr_height     = config_manager.getApproachThrHeight();
    approach_params.recent_w       = config_manager.getApproachRecentW();
    approach_params.score_thr      = config_manager.getApproachScoreThr();
    approach_params.confirm        = config_manager.getApproachConfirm();
    approach_params.exit_score_thr = config_manager.getApproachExitScoreThr();
    approach_params.exit_confirm   = config_manager.getApproachExitConfirm();

    // filter 字符串 -> 枚举；未识别的取值回落到 none 并告警（避免静默改变行为）
    const std::string    filter_name = config_manager.getApproachFilterMode();
    approach::FilterMode filter_mode = approach::parseFilterMode(filter_name);
    if (filter_mode == approach::FilterMode::NONE && filter_name != "none") {
        APP_WARN("Unknown approach filter '{}', fallback to none", filter_name);
    }
    motion_state_engine_.configureApproach(approach_params, filter_mode,
                                           config_manager.isApproachEnabled());

    // 是否加载深度模型由 config 的 depth.enabled 控制
    if (config_manager.isDepthEnabled()) {
        if (config_manager.getDepthModelType() == "lite_mono") {
            depth_model_.init(config_manager.getDepthModelPath(), frame_meta.img_w,
                              frame_meta.img_h, is_normalize, config_manager.isUseGPU());
        } else if (config_manager.getDepthModelType() == "yolo_depth") {
            yolo_depth_model_.init(config_manager.getDepthModelPath(), frame_meta.img_w,
                                   frame_meta.img_h, config_manager.isUseGPU());
        } else {
            APP_ERROR("Unsupported depth model type: {}", config_manager.getDepthModelType());
        }
    }
    detector_.init(config_manager.getYoloModelPath(), frame_meta.img_w, frame_meta.img_h,
                   config_manager.getYoloNmsThresh(), config_manager.getYoloConfThresh(), 80,
                   config_manager.isUseGPU());

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

void Pipeline::process(FrameInputContext &  frame_input_context,
                       InferOutputContext & infer_output_context) {
    // 检测模型（YoloDetectModel detector_）同步推理耗时
    const auto t_detect_begin = std::chrono::steady_clock::now();
    detector_.runInference(frame_input_context, infer_output_context);
    const auto t_detect_end = std::chrono::steady_clock::now();

    // 深度模型推理耗时（注意：同步路径用的是 depth_model_，重叠路径才用 yolo_depth_model_）
    double depth_ms = 0.0;
    if (depth_enabled_) {
        const auto t_depth_begin = std::chrono::steady_clock::now();
        depth_model_.runInference(frame_input_context, infer_output_context);
        const auto t_depth_end = std::chrono::steady_clock::now();
        depth_ms               = durationMs(t_depth_begin, t_depth_end);
    }

    updateTracker(infer_output_context);
    updateMotionStates(frame_input_context, infer_output_context);

    if (depth_enabled_) {
        APP_INFO("infer latency frame {}: yolo_detect = {:.2f} ms, depth_model_ = {:.2f} ms",
                 frame_input_context.frame_id, durationMs(t_detect_begin, t_detect_end), depth_ms);
    } else {
        APP_INFO("infer latency frame {}: yolo_detect = {:.2f} ms", frame_input_context.frame_id,
                 durationMs(t_detect_begin, t_detect_end));
    }
}

void Pipeline::processOverlap(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context) {
    // yolo_detect_model_ (detector_)：从发起异步推理到结果可取
    detector_.runInferenceAsync(frame_input_context);

    // yolo_depth_model_：从发起异步推理到结果可取（与检测在不同 stream 上可重叠执行）
    if (depth_enabled_) {
        // depth_model_.runInferenceAsync(frame_input_context);
        yolo_depth_model_.runInferenceAsync(frame_input_context);
    }
    detector_.getInferOutputResult(infer_output_context);
    updateTracker(infer_output_context);
    if (depth_enabled_) {
        // depth_model_.getInferOutputResult(infer_output_context);
        yolo_depth_model_.getInferOutputResult(infer_output_context);
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
        const auto dims = yolo_depth_model_.getInputDims();
        if (dims.size() >= 4 && dims[2] > 0 && dims[3] > 0) {
            cv::Mat raw_depth_map(dims[2], dims[3], CV_32FC1,
                                  infer_output_context.depth_raw_infer_out.data());
            cv::resize(raw_depth_map, depth_metric,
                       cv::Size(frame_input_context.meta.img_w, frame_input_context.meta.img_h));
        }
    }

    // ---- 快速靠近（approach）检测：逐 track 独立判定，人/车不合并 ----
    // 参与判定的类别只有 bicycle/car/motorcycle/bus/truck（person 只跟踪不判定）。
    // 每个 track 用自己的框高 + 框内深度送进 updateApproachState，
    // 逐帧因果、可实时触发；该判定即危险依据（不再有 TTC）。
    for (auto & track : infer_output_context.tracked_objects) {
        approach::ApproachState approach_state;
        bool                    has_approach = false;
        float                   depth        = 0.0f;

        if (isApproachClass(track.class_id_)) {
            // 框内鲁棒深度估计（深度不可用时为 0，检测器视为无效值）
            if (!depth_metric.empty()) {
                depth           = motion_state_engine_.computeMeanDepth(depth_metric, track.tlwh_);
                track.distance_ = depth;
            }
            // 目标框高度 + 框内深度 -> 逐帧接近判定（track_id 即判定状态索引）
            approach_state = motion_state_engine_.updateApproachState(
                track.track_id_, track.tlwh_[3], depth, frame_input_context.timestamp);
            has_approach = true;
        }

        // 记录本帧判定结果：只有“快速靠近”一路（是否危险即看 approach_alarm）。
        // person 等不参与判定的类别全为 false/0
        MotionStateInfoRecord motion;
        if (has_approach) {
            motion.approach_alarm       = approach_state.alarm;
            motion.approach_score       = approach_state.score;
            motion.approach_depth_score = approach_state.depth_score;
            motion.approach_scale_score = approach_state.scale_score;
        }
        infer_output_context.motion_records.emplace(track.track_id_, motion);

        // track_log：每个目标框一行（与 Python 侧 CSV 口径一致）；
        // 参与判定的类别复用上面那次深度采样，其它类别单独采样
        if (track_log_enabled_) {
            float raw_depth = depth;
            if (!has_approach && !depth_metric.empty()) {
                raw_depth = motion_state_engine_.computeMeanDepth(depth_metric, track.tlwh_);
            }
            track_log_data_[track.track_id_].push_back(
                { frame_input_context.frame_id, track.class_id_, track.tlwh_[0], track.tlwh_[2],
                  track.tlwh_[1], track.tlwh_[3], track.tlwh_[2] * track.tlwh_[3], raw_depth });
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
