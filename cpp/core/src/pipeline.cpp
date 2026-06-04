#include "pipeline.h"

#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "logger_manager.h"
#include "motion_state_engine.h"
#include "public.h"

#include <array>

Pipeline::Pipeline(const ConfigManager & config_manager, FrameMeta frame_meta) :
    config_manager_(config_manager),
    tracker_(30, 30),  // 假设fps=30，或从config读取
    motion_state_engine_(config_manager_.getMotionVelocityThreshold(),
                         config_manager_.getMotionAccelerationThreshold(),
                         config_manager_.getKfProcessNoiseCov(),
                         config_manager_.getKfMeasurementNoiseCov()) {
    bool is_normalize = false;
    if (config_manager_.getDepthEnginePath().find("depth_anything") != std::string::npos) {
        is_normalize = true;
    } else if (config_manager_.getDepthEnginePath().find("lite") != std::string::npos) {
        is_normalize = false;
    }
    depth_model_.init(config_manager_.getDepthEnginePath(), frame_meta.img_w, frame_meta.img_h,
                      is_normalize);
    detector_.init(config_manager_.getYoloEnginePath(), frame_meta.img_w, frame_meta.img_h,
                   config_manager_.getYoloNmsThresh(), config_manager_.getYoloConfThresh(), 80);
}

void Pipeline::init() {}

// 同步
void Pipeline::process(FrameInputContext &  frame_input_context,
                       InferOutputContext & infer_output_context) {
    // bool do_depth = (!has_cached_depth_) ||
    //                 ((frame_input_context.frame_id - 1) % config_manager_.getDepthInterval() == 0);

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
    const std::vector<int> detect_input_dims   = detector_.getInputDims();
    const int              MAX_NUM_OUTPUT_BBOX = detector_.MAX_NUM_OUTPUT_BBOX;
    const int              NUM_BOX_ELEMENT     = detector_.NUM_BOX_ELEMENT;
    detector_.setFeedInferOutputCallback([detect_input_dims, MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT](
                                             FrameInputContext &  frame_input_context,
                                             InferOutputContext & infer_output_context,
                                             void *               detect_output_data) {
        APP_INFO("detector_.setFeedInferOutputCallback");
        float *                detect_output_data_float = static_cast<float *>(detect_output_data);
        std::vector<Detection> vDetections;
        int count = std::min(static_cast<int>(detect_output_data_float[0]), MAX_NUM_OUTPUT_BBOX);

        for (int i = 0; i < count; i++) {
            int pos      = 1 + i * NUM_BOX_ELEMENT;
            int keepFlag = static_cast<int>(detect_output_data_float[pos + 6]);

            if (keepFlag == 1) {
                Detection det;
                memcpy(det.bbox.data(), &detect_output_data_float[pos], 4 * sizeof(float));
                det.conf    = detect_output_data_float[pos + 4];
                det.classId = static_cast<int>(detect_output_data_float[pos + 5]);
                vDetections.push_back(det);
            }
        }
        int input_w = detect_input_dims[3];
        int input_h = detect_input_dims[2];

        // 缩放 bbox 到原始图像尺寸
        for (auto & det : vDetections) {
            std::array<float, 4> bbox  = det.bbox;
            float                r_w   = input_w / (frame_input_context.raw_img.cols * 1.0);
            float                r_h   = input_h / (frame_input_context.raw_img.rows * 1.0);
            float                r     = std::min(r_w, r_h);
            float                pad_h = (input_h - r * frame_input_context.raw_img.rows) / 2;
            float                pad_w = (input_w - r * frame_input_context.raw_img.cols) / 2;

            bbox[0] = (bbox[0] - pad_w) / r;
            bbox[1] = (bbox[1] - pad_h) / r;
            bbox[2] = (bbox[2] - pad_w) / r;
            bbox[3] = (bbox[3] - pad_h) / r;
        }
        infer_output_context.detections = vDetections;
    });

    depth_model_.setFeedInferOutputCallback([](FrameInputContext &  frame_input_context,
                                               InferOutputContext & infer_output_context,
                                               void *               depth_output_data) {
        DepthInferOutputSet * depth_infer_set =
            static_cast<DepthInferOutputSet *>(depth_output_data);

        APP_INFO("test depth_model_.setFeedInferOutputCallback depth");
        infer_output_context.result_depth =
            cv::Mat(frame_input_context.meta.img_h, frame_input_context.meta.img_w, CV_8UC1,
                    depth_infer_set->depth_output_data);
        infer_output_context.depth_vis =
            cv::Mat(frame_input_context.meta.img_h, frame_input_context.meta.img_w, CV_8UC3,
                    depth_infer_set->depth_colormap_data);
        delete depth_infer_set;
    });

    detector_.runInferenceAsync(frame_input_context, infer_output_context);
    depth_model_.runInferenceAsync(frame_input_context, infer_output_context);

    detector_.waitAsync();
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

    depth_model_.waitAsync();

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
