#pragma once
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

class STrack;
class Detection;

enum MotionState {
    INVAILD   = 0,
    UNKNOWN   = 1,
    STABLE    = 2,
    APPROACH  = 3,
    MOVE_AWAY = 4,
    ACCELE    = 5,
    DECELE    = 6,
    CONSTANT  = 7
};

struct MotionStateInfoRecord {
    // ttc / ttc_danger 带默认值：现有 3 参构造调用点无需改动（增量兼容）
    MotionStateInfoRecord(MotionState state_vec,
                          MotionState state_acc,
                          float       velocity,
                          float       ttc        = -1.0f,
                          bool        ttc_danger = false) :
        state_vec(state_vec),
        state_acc(state_acc),
        velocity(velocity),
        ttc(ttc),
        ttc_danger(ttc_danger) {}

    MotionState state_vec;
    MotionState state_acc;
    float       velocity;
    float       ttc;  // 碰撞时间（秒）；-1 表示无效/静止/远离
    bool ttc_danger;  // TTC 危险报警（引擎内延迟阻塞判定）；false 表示无报警
};

struct InferOutputContext {
    std::vector<Detection>                         detections;
    std::vector<STrack>                            tracked_objects;
    std::vector<float>                             depth_raw_infer_out;
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
        img_size = meta.img_h * meta.img_w * 3;
    }

    void setFrameID(int id) { frame_id = id; }

    int                    frame_id;
    FrameMeta              meta;
    double                 timestamp;
    unique_ptr_cuda<uchar> d_raw_img_;
    cv::Mat                raw_img;
    size_t                 img_size;
};
