#pragma once
#include "public.h"

#include <memory.h>
#include <opencv2/core/hal/interface.h>

#include <cstddef>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

enum FrameSource { VIDEO, CAMERA };

struct FrameMeta {
    FrameMeta() = default;

    FrameMeta(int img_w, int img_h, double fps, FrameSource frame_source) :
        img_w(img_w),
        img_h(img_h),
        fps(fps),
        frame_source(frame_source) {}

    int    img_w;
    int    img_h;
    double fps;

    FrameSource frame_source;
};

struct MotionStateInfoRecord;
class STrack;
class Detection;

struct InferOutputContext {
    std::vector<Detection>                         detections;
    std::vector<STrack>                            tracked_objects;
    cv::Mat                                        result_depth;
    cv::Mat                                        depth_vis;
    std::unordered_map<int, MotionStateInfoRecord> motion_records;
};

struct FrameInputContext {
    FrameInputContext(int frame_id, FrameMeta meta) : frame_id(frame_id), meta(meta) {
        if (meta.frame_source == FrameSource::VIDEO) {
            timestamp = (meta.fps > 0.0) ? (frame_id / meta.fps) : 0.0;
        } else if (meta.frame_source == FrameSource::CAMERA) {
            timestamp =
                std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                    .count();
        }
        img_size                 = meta.img_h * meta.img_w * 3;
        void *      ptr          = nullptr;
        int         device_count = 0;
        cudaError_t error        = cudaGetDeviceCount(&device_count);
        if (error == cudaSuccess && device_count > 0) {
            CHECK_CUDA(cudaMalloc(&ptr, img_size));
            d_raw_img_.reset(static_cast<uchar *>(ptr));
        }
    }

    void setFrameID(int id) { frame_id = id; }

    int                    frame_id;
    FrameMeta              meta;
    double                 timestamp;
    unique_ptr_cuda<uchar> d_raw_img_;
    cv::Mat                raw_img;
    size_t                 img_size;
};
