#include "pipeline.h"

#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "logger_manager.h"
#include "motion_state_engine.h"
#include "public.h"

#include <array>

Pipeline::Pipeline(ConfigManager & config_manager, FrameMeta frame_meta) :
    config_manager_(config_manager),
    tracker_(30, 30),  // 假设fps=30，或从config读取
    motion_state_engine_(config_manager_.getMotionVelocityThreshold(),
                         config_manager_.getMotionAccelerationThreshold(),
                         config_manager_.getKfProcessNoiseCov(),
                         config_manager_.getKfMeasurementNoiseCov()) {
    bool is_normalize = false;
    // depth_model_.init(config_manager_.getDepthEnginePath(), frame_meta.img_w, frame_meta.img_h,
    //                   is_normalize);
    // detector_.init(config_manager_.getYoloEnginePath(), frame_meta.img_w, frame_meta.img_h,
    //                config_manager_.getYoloNmsThresh(), config_manager_.getYoloConfThresh(), 80);

    depth_model_.init(config_manager_.getDepthModelPath(), frame_meta.img_w, frame_meta.img_h,
                      is_normalize, config_manager_.isUseGPU());
    detector_.init(config_manager_.getYoloModelPath(), frame_meta.img_w, frame_meta.img_h,
                   config_manager_.getYoloNmsThresh(), config_manager_.getYoloConfThresh(), 80,
                   config_manager_.isUseGPU());
}

void Pipeline::init() {}

// 同步
void Pipeline::process(FrameInputContext &  frame_input_context,
                       InferOutputContext & infer_output_context) {
    detector_.runInference(frame_input_context, infer_output_context);
    depth_model_.runInference(frame_input_context, infer_output_context);

    std::vector<Detection> res = infer_output_context.detections;
    std::vector<Object>    objects;
    for (size_t j = 0; j < res.size(); j++) {
        if (isTrackingClass(res[j].classId)) {
            cv::Rect_<float> rect(res[j].bbox[0], res[j].bbox[1], (res[j].bbox[2] - res[j].bbox[0]),
                                  (res[j].bbox[3] - res[j].bbox[1]));
            objects.push_back({ rect, res[j].classId, res[j].conf });
        }
    }
    infer_output_context.tracked_objects = tracker_.update(objects);
    this->postProcess(frame_input_context, infer_output_context);

    // if (do_depth) {
    //     auto depth_infer_result = depth_model_.runInference(frame_input_context.raw_img, );

    //     infer_output_context.result_depth = depth_infer_result.first;
    //     infer_output_context.depth_vis    = depth_infer_result.second;
    //     has_cached_depth_                 = true;
    // } else {
    //     infer_output_context.result_depth = cached_depth_;
    //     infer_output_context.depth_vis    = cached_depth_vis_;
    // }

    // std::vector<Detection> res = detector_.runInference(frame_input_context.raw_img);
    // postProcess(frame_input_context, infer_output_context);
}

void Pipeline::processOverlap(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context) {
    detector_.runInferenceAsync(frame_input_context);
    depth_model_.runInferenceAsync(frame_input_context);

    detector_.getInferOutputResult(infer_output_context);
    {
        nvtx3::scoped_range    byte_tracker_scope("byte tracker process");
        std::vector<Detection> res = infer_output_context.detections;
        std::vector<Object>    objects;
        for (size_t j = 0; j < res.size(); j++) {
            if (isTrackingClass(res[j].classId)) {
                cv::Rect_<float> rect(res[j].bbox[0], res[j].bbox[1],
                                      (res[j].bbox[2] - res[j].bbox[0]),
                                      (res[j].bbox[3] - res[j].bbox[1]));
                objects.push_back({ rect, res[j].classId, res[j].conf });
            }
        }
        infer_output_context.tracked_objects = tracker_.update(objects);
    }

    depth_model_.getInferOutputResult(infer_output_context);

    this->postProcess(frame_input_context, infer_output_context);
}

void Pipeline::postProcess(FrameInputContext &  frame_input_context,
                           InferOutputContext & infer_output_context) {
    nvtx3::scoped_range tracker_scope("pipeline postProcess");
    infer_output_context.motion_records.clear();
    for (int i = 0; i < infer_output_context.tracked_objects.size(); i++) {
        if (infer_output_context.tracked_objects[i].tlwh_[2] *
                infer_output_context.tracked_objects[i].tlwh_[3] <=
            20) {
            continue;
        }
        int track_id = infer_output_context.tracked_objects[i].track_id_;

        float current_depth = motion_state_engine_.getObjectDepth(
            infer_output_context.result_depth, infer_output_context.tracked_objects[i],
            frame_input_context.raw_img.size());

        infer_output_context.motion_records.insert(
            { track_id, motion_state_engine_.computeMotionState(track_id, current_depth,
                                                                frame_input_context.timestamp) });
    }
}
