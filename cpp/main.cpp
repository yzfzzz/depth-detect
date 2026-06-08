
#include "config_manager.h"
#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"
#include "scope_timer.h"
#include "visual_manager.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdio>
#include <functional>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <string>
#include <utility>
#include <vector>

cv::Mat drawOneFrame(FrameInputContext &            frame_input_context,
                     InferOutputContext &           infer_output_context,
                     const ConfigManager &          config_manager,
                     DrawingManager &               drawing_manager,
                     std::function<cv::Scalar(int)> get_color_func,
                     int                            total_us) {
    if (infer_output_context.tracked_objects.size() <= 0) {
        APP_WARN("tracked_objects size is 0, skip drawing!");
    }
    for (int i = 0; i < infer_output_context.tracked_objects.size(); i++) {
        auto & track = infer_output_context.tracked_objects[i];
        if (track.tlwh_[2] * track.tlwh_[3] <= 20) {
            continue;
        }

        auto it = infer_output_context.motion_records.find(track.track_id_);
        if (it != infer_output_context.motion_records.end()) {
#if defined(ENABLE_TIMER)
            DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(
                "6.Drawing Manager", drawing_manager, drawTrackedObject,
                frame_input_context.raw_img, track, it->second, get_color_func(track.track_id_));
#else
            drawing_manager.drawTrackedObject(frame_input_context.raw_img, track, it->second,
                                              get_color_func(track.track_id_));
#endif
        }
    }
    // FPS
    int show_fps = (total_us > 0) ? (frame_input_context.frame_id * 1000000LL / total_us) : 0;
    // 全局信息
    drawing_manager.drawGlobalInfo(frame_input_context.raw_img, frame_input_context.frame_id,
                                   show_fps, infer_output_context.tracked_objects.size());

    // 上下拼接
    cv::Mat out_frame = drawing_manager.concatenateFrames(frame_input_context.raw_img,
                                                          infer_output_context.depth_vis);
    return out_frame;
}

int run(char * video_path, char * config_path) {
    // 读取配置文件 - 使用单例模式
    ConfigManager & config_manager = ConfigManager::getInstance(config_path);
    // 初始化日志系统
    LoggerManager::getInstance(config_manager);
    APP_INFO("Application started with video: {}", std::string(video_path));
    // 文件读写，落盘保存, 以及视频读取（包括模拟相机延迟）
    IOManager      io_manager(config_manager);
    FrameMeta      frame_meta = io_manager.Init(video_path);
    // 推理流水线（负责目标检测、深度估计、跟踪、运动状态判断等核心功能）
    Pipeline       pipeline(config_manager, frame_meta);
    // 绘制管理器（负责绘制结果）
    DrawingManager drawing_manager(V_CLASS_NAMES);
    // 显示管理器（负责窗口管理、显示、鼠标点击等）
    DisplayManager display_manager(config_manager, "Detection Result",
                                   cv::Size(frame_meta.img_w, frame_meta.img_h * 2));

    int                num_frames = 0;
    double             total_us   = 0;
    FrameInputContext  frame_input_context(num_frames, frame_meta);
    InferOutputContext infer_output_context;
    while (true) {
        frame_input_context.setFrameID(num_frames);
#if defined(ENABLE_TIMER)
        if (!DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("1.Cap Read", io_manager, readNextFrame,
                                                    frame_input_context, false) ||
            frame_input_context.raw_img.empty()) {
            break;
        }
        // 执行推理流水线
        std::string name = "Infer Pipeline";
        // DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, pipeline, process, frame_input_context, infer_output_context);
        if (config_manager.isOverlapEnabled()) {
            DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, pipeline, processOverlap,
                                                   frame_input_context, infer_output_context);
        } else {
            DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, pipeline, process, frame_input_context,
                                                   infer_output_context);
        }
        total_us += ScopedTimer::GetScopedTimers()[name].back();  // 获取刚刚这次推理的耗时
#else
        if (!io_manager.readNextFrame(frame_input_context, false) ||
            frame_input_context.raw_img.empty()) {
            break;
        }

        if (config_manager.isOverlapEnabled()) {
            pipeline.processOverlap(frame_input_context, infer_output_context);
        } else {
            pipeline.process(frame_input_context, infer_output_context);
        }

#endif
        num_frames++;
        if (num_frames % 100 == 0) {
            APP_INFO("Processing frame {} ({:.2f} fps)", num_frames,
                     (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0));
        }

        // 画图
        cv::Mat out_frame = drawOneFrame(
            frame_input_context, infer_output_context, config_manager, drawing_manager,
            [&pipeline](int idx) { return pipeline.getColor(idx); }, total_us);
        // 保存结果
        io_manager.saveFrame(out_frame, num_frames);

        // 显示图像（通过 DisplayManager）
        display_manager.updateData(infer_output_context.tracked_objects,
                                   infer_output_context.result_depth);
        if (display_manager.isEnabled()) {
            display_manager.show(out_frame);
            char c      = display_manager.waitKey(1);
            int  result = display_manager.handleKey(c);
            if (result == Key_Input::ESC) {  // 用户按下 ESC 键退出
                break;
            }
        }
    }

#if defined(ENABLE_TIMER)
    APP_INFO("==========Summary===========");
    for (auto & kv : ScopedTimer::GetScopedTimers()) {
        double avg = calculateAverage(kv.second);
        double p95 = calculatePercentile(kv.second, 95.0);
        double p99 = calculatePercentile(kv.second, 99.0);
        APP_INFO("[{}]: avg = {:.2f} ms, P95 = {:.2f} ms, P99 = {:.2f} ms (frame)", kv.first, avg,
                 p95, p99);
    }
#endif

    return 0;
}

int main(int argc, char * argv[]) {
    if (argc != 3) {
        APP_ERROR("arguments not right!");
        APP_ERROR("Usage: ./main [video path] [config yaml path]");
        APP_ERROR("Example: ./main ./videos/demo.mp4 ./config.yaml");
        return -1;
    }

    return run(argv[1], argv[2]);
}
