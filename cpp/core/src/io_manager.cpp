#include "io_manager.h"

#include "frame.h"
#include "logger_manager.h"
#include "public.h"

#include <cstdlib>  // For system()
#include <iostream>
#include <ostream>
#include <thread>

IOManager::IOManager(const ConfigManager & config_manager) :
    save_mode_(config_manager.getSaveMode()),
    out_dir_(config_manager.getOutDir()) {}

FrameMeta IOManager::Init(const std::string & video_path) {
    // 如果需要保存图片，检查目标文件夹并创建
    if (save_mode_ != "none" && out_dir_ != "" && !dirExists(out_dir_)) {
        makeDir(out_dir_);
    }
    bool flag = openVideoSource(video_path);
    if (!flag) {
        APP_ERROR("Failed to open video source: {}", video_path);
    }
    FrameMeta frame_meta = getVideoFrameMeta();

    // 如果需要保存视频，初始化 VideoWriter
    if (save_mode_ == "video" || save_mode_ == "both") {
        std::string video_save_path = out_dir_ + "/result.mp4";  // 最好也放进输出目录
        video_writer_.open(video_save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           frame_meta.fps, cv::Size(frame_meta.img_w, frame_meta.img_h));

        if (!video_writer_.isOpened()) {
            APP_ERROR("Failed to initialize VideoWriter at {}", video_save_path);
        }
    }
    return frame_meta;
}

IOManager::~IOManager() {
    if (video_writer_.isOpened()) {
        video_writer_.release();
    }
    closeVideoSource();
}

void IOManager::saveFrame(const cv::Mat & frame, int num_frames) {
    if (save_mode_ == "images" || save_mode_ == "both") {
        std::string save_path = out_dir_ + "/frame_" + std::to_string(num_frames) + ".jpg";
        cv::imwrite(save_path, frame);
    }

    if (save_mode_ == "video" || save_mode_ == "both") {
        if (video_writer_.isOpened()) {
            video_writer_.write(frame);
        }
    }
}

bool IOManager::dirExists(const std::string & path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

void IOManager::makeDir(const std::string & path) {
    std::string cmd = "mkdir -p " + path;
    int         ret = system(cmd.c_str());
    if (ret != 0) {
        APP_WARN("Could not execute mkdir completely.");
    }
}

bool IOManager::openVideoSource(const std::string & video_path) {
    video_capture_.open(video_path);
    if (!video_capture_.isOpened()) {
        APP_ERROR("Failed to open video: {}", video_path);
        return false;
    }
    is_first_frame_ = true;
    double fps      = video_capture_.get(cv::CAP_PROP_FPS);
    if (fps > 0) {
        frame_interval_ms_ = 1000.0 / fps;
    }
    long total_frames_num = static_cast<long>(video_capture_.get(cv::CAP_PROP_FRAME_COUNT));
    APP_INFO("Total frames: {}", total_frames_num);
    return true;
}

void IOManager::closeVideoSource() {
    if (video_capture_.isOpened()) {
        video_capture_.release();
    }
}

FrameMeta IOManager::getVideoFrameMeta() const {
    if (!video_capture_.isOpened()) {
        APP_WARN("Video source not opened, returning default FrameMeta");
        return FrameMeta(0, 0, 0, FrameSource::VIDEO);
    }
    return FrameMeta(video_capture_.get(cv::CAP_PROP_FRAME_WIDTH),
                     video_capture_.get(cv::CAP_PROP_FRAME_HEIGHT),
                     video_capture_.get(cv::CAP_PROP_FPS), FrameSource::VIDEO);
}

bool IOManager::readNextFrame(FrameInputContext & frame_input_context, bool simulate_delay) {
    if (!video_capture_.isOpened()) {
        return false;
    }

    // 第一帧或者不模拟延迟时，直接读取
    if (is_first_frame_ || !simulate_delay) {
        is_first_frame_ = false;
    } else {
        // 计算上一帧的实际处理耗时
        auto frame_process_start = std::chrono::steady_clock::now();
        auto elapsed_ms =
            std::chrono::duration<double, std::milli>(frame_process_start - last_frame_start_time_)
                .count();

        // 只有当有有效耗时和有效帧间隔时才计算跳帧
        if (frame_interval_ms_ > 0) {
            int frames_to_skip = static_cast<int>(elapsed_ms / frame_interval_ms_) - 1;

            // 跳过相应的帧（模拟相机延迟）
            for (int skip = 0; skip < frames_to_skip; skip++) {
                cv::Mat dummy;
                if (!video_capture_.read(dummy)) {
                    return false;
                }
            }
        }
    }
    // 读取处理用的当前帧
    bool        result       = video_capture_.read(frame_input_context.raw_img);
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    if (result && error == cudaSuccess && device_count > 0) {
        CHECK_CUDA(cudaMemcpy(frame_input_context.d_raw_img_.get(),
                              frame_input_context.raw_img.data, frame_input_context.img_size,
                              cudaMemcpyHostToDevice));
    }
    // 更新下一帧的处理开始时间
    last_frame_start_time_ = std::chrono::steady_clock::now();

    return result;
}
