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

// 目标的运动/危险判定结果：当前只保留“快速靠近”（approach）一路，判定逻辑见 motion_state_engine.h
// （基线 + 累计变化率 + 近期趋势 + 进出双边去抖）。原来的 TTC/运动状态一路已移除：
// 危险与否只看 approach_alarm，快速靠近就上报危险（见 DangerAlertHandler）并画红框。
struct MotionStateInfoRecord {
    bool  approach_alarm       = false;  // 接近报警（已通过进入/退出双边去抖）
    float approach_score       = 0.0f;   // 融合分数 = 0.4 * 深度分 + 0.6 * 尺度分
    float approach_depth_score = 0.0f;   // 深度分（相对基线的累计降幅 / thr_depth）
    float approach_scale_score = 0.0f;   // 尺度分（相对基线的框高累计增幅 / thr_height）
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
