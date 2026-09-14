#include "io_manager.h"

#include "frame.h"
#include "logger_manager.h"
#include "public.h"

#include <algorithm>  // std::all_of
#include <cctype>     // std::isdigit
#include <cstdlib>    // For system()
#include <thread>     // std::this_thread::sleep_for（实时节奏模拟）

IOManager::IOManager(const ConfigManager & config_manager) :
    IOManager(config_manager.getSaveMode(),
              config_manager.getOutDir(),
              config_manager.getSendTcpIp(),
              config_manager.getSendTcpPort(),
              config_manager.isSendTcpEnabled(),
              config_manager.getCameraWidth(),
              config_manager.getCameraHeight(),
              config_manager.getCameraFps(),
              config_manager.getSimulateFps(),
              config_manager.isSimulateDelayEnabled()) {}

IOManager::IOManager(std::string save_mode,
                     std::string out_dir,
                     std::string send_tcp_ip,
                     int         send_tcp_port,
                     bool        send_tcp_enabled,
                     int         camera_width,
                     int         camera_height,
                     int         camera_fps,
                     int         simulate_fps,
                     bool        simulate_delay) :
    save_mode_(std::move(save_mode)),
    out_dir_(std::move(out_dir)),
    send_tcp_ip_(std::move(send_tcp_ip)),
    send_tcp_port_(send_tcp_port),
    send_tcp_enabled_(send_tcp_enabled),
    camera_width_(camera_width),
    camera_height_(camera_height),
    camera_fps_(camera_fps),
    simulate_fps_(simulate_fps),
    simulate_delay_(simulate_delay) {}

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

    // 如果需要保存视频，仅记录路径与写盘 fps；VideoWriter 在首帧保存时懒初始化，
    // 因为写盘尺寸必须等于实际落盘帧的尺寸（深度启用时是上下拼接的 2 倍高），
    // 用 meta 尺寸初始化会因尺寸不匹配导致写帧静默失败、产出损坏的空 mp4
    if (save_mode_ == "video" || save_mode_ == "both") {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto tm   = std::localtime(&time);

        std::ostringstream oss;
        oss << "result_" << std::put_time(tm, "%Y%m%d_%H%M%S") << ".mp4";
        video_save_path_ = out_dir_ + "/" + oss.str();
        // 写盘 fps = 实际产出帧率：模拟节奏下每墙钟秒只产出 simulate_fps 帧，
        // 若仍按源视频 fps 写文件头，回放会被等比加速（帧数少了一半，播放速度却不变）
        writer_fps_      = frame_meta.fps;
        if (simulate_delay_ && simulate_fps_ > 0) {
            writer_fps_ = simulate_fps_;
        }
        if (writer_fps_ <= 0) {
            writer_fps_ = 30.0;
        }
    }
    if (send_tcp_enabled_) {
        APP_INFO("TCP sending is enabled. Will send JSON data to {}:{}", send_tcp_ip_,
                 send_tcp_port_);
        json_sender_ptr_   = std::make_unique<JsonSender>(send_tcp_ip_, send_tcp_port_);
        is_json_sender_ok_ = (json_sender_ptr_->get_fd() >= 0);
        if (!is_json_sender_ok_) {
            APP_WARN(
                "Failed to initialize JsonSender for {}:{}, we will not send json data to server.",
                send_tcp_ip_, send_tcp_port_);
        } else {
            // 对端处理慢或 TCP 发送缓冲区满了，会导致发送失败，因此需要设置非阻塞，直接放弃发送防止卡死
            json_sender_ptr_->set_nonblocking();
        }
    } else {
        APP_WARN("TCP sending is disabled.");
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
        // 首帧懒初始化：写盘尺寸取实际落盘帧尺寸（深度启用时为上下拼接的 2 倍高）
        if (!video_writer_.isOpened() && !video_save_path_.empty()) {
            video_writer_.open(video_save_path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               writer_fps_, frame.size());
            if (!video_writer_.isOpened()) {
                APP_ERROR("Failed to initialize VideoWriter at {}", video_save_path_);
                video_save_path_.clear();  // 防止之后每帧重复尝试与报错
            }
        }
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
    // 通过 shell 创建目录（跨平台兼容性注意：Windows 需改为 _mkdir）
    // TODO：考虑跨平台兼容性
    std::string cmd = "mkdir -p " + path;
    int         ret = system(cmd.c_str());
    if (ret != 0) {
        APP_WARN("Could not execute mkdir completely.");
    }
}

bool IOManager::openVideoSource(const std::string & video_path) {
    // 纯数字字符串视为相机索引（USB 相机）
    const bool is_camera_index =
        !video_path.empty() && std::all_of(video_path.begin(), video_path.end(),
                                           [](unsigned char c) { return std::isdigit(c) != 0; });
    if (is_camera_index) {
        int  camera_index = std::stoi(video_path);
        // 按 config 的 camera 段配置打开相机并设置宽高/帧率
        auto open_camera  = [&]() {
            video_capture_.open(camera_index, cv::CAP_V4L2);
            video_capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            video_capture_.set(cv::CAP_PROP_FRAME_WIDTH, camera_width_);
            video_capture_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height_);
            if (camera_fps_ > 0) {
                video_capture_.set(cv::CAP_PROP_FPS, camera_fps_);
            }
        };
        open_camera();
        // 部分相机需要先设置 FOURCC/分辨率后再打开才能生效
        if (!video_capture_.isOpened() || video_capture_.get(cv::CAP_PROP_FOURCC) !=
                                              cv::VideoWriter::fourcc('M', 'J', 'P', 'G')) {
            video_capture_.release();
            open_camera();
        }
    } else {
        video_capture_.open(video_path);
    }
    if (!video_capture_.isOpened()) {
        APP_ERROR("Failed to open video: {}", video_path);
        return false;
    }
    is_first_frame_   = true;
    skip_accumulator_ = 0.0;
    double fps        = video_capture_.get(cv::CAP_PROP_FPS);
    // 视频自身帧间隔：跳帧对齐的基准（墙钟走 1ms，视频前进多少由它决定）
    video_frame_ms_   = (fps > 0) ? 1000.0 / fps : 0.0;
    if (simulate_fps_ > 0) {
        // 模拟现场帧率优先：不依赖视频自身的 fps 元数据
        frame_interval_ms_ = 1000.0 / simulate_fps_;
        APP_INFO(
            "Simulate delay enabled: {} fps (tick {:.1f} ms), video fps = {:.1f}, "
            "video will advance 1:1 with wall clock",
            simulate_fps_, frame_interval_ms_, fps);
    } else if (fps > 0) {
        frame_interval_ms_ = 1000.0 / fps;
        APP_INFO("Simulate delay enabled: follow video fps = {:.1f} (tick {:.1f} ms)", fps,
                 frame_interval_ms_);
    } else {
        // 视频无有效 fps 元数据时退化为不限速（frame_interval_ms_ <= 0 时不做节奏控制）
        frame_interval_ms_ = 0.0;
        APP_WARN("Video fps metadata invalid, simulate delay disabled");
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
    // TODO: 异步读取
    if (!video_capture_.isOpened()) {
        return false;
    }

    if (is_first_frame_ || !simulate_delay || frame_interval_ms_ <= 0) {
        is_first_frame_ = false;
        last_tick_time_ = std::chrono::steady_clock::now();
    } else {
        // 实时节拍模拟: 模拟端侧设备低帧率运行
        // 从相机取流，每个 tick 只保留这段时间里视频产出的最新一帧，过时帧直接丢弃
        std::this_thread::sleep_until(
            last_tick_time_ + std::chrono::duration<double, std::milli>(frame_interval_ms_));
        auto   now       = std::chrono::steady_clock::now();
        double window_ms = std::chrono::duration<double, std::milli>(now - last_tick_time_).count();
        last_tick_time_  = now;
        if (video_frame_ms_ > 0) {
            // 窗口内视频产出的帧数；小数进累积器跨 tick 累计，保证总量一致
            skip_accumulator_ += window_ms / video_frame_ms_;
            int frames_to_skip = static_cast<int>(skip_accumulator_) - 1;
            if (frames_to_skip > 0) {
                skip_accumulator_ -= frames_to_skip + 1;
                for (int skip = 0; skip < frames_to_skip; skip++) {
                    cv::Mat dummy;
                    if (!video_capture_.read(dummy)) {
                        return false;
                    }
                }
            } else {
                skip_accumulator_ -= 1.0;
            }
        }
    }
    // 读取当前帧并同步拷贝到 GPU，供 CUDA 预处理使用
    bool result = video_capture_.read(frame_input_context.raw_img);

    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    if (result && error == cudaSuccess && device_count > 0) {
        if (!frame_input_context.d_raw_img_) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&ptr, frame_input_context.img_size));
            frame_input_context.d_raw_img_.reset(static_cast<uchar *>(ptr));
        }
        CHECK_CUDA(cudaMemcpy(frame_input_context.d_raw_img_.get(),
                              frame_input_context.raw_img.data, frame_input_context.img_size,
                              cudaMemcpyHostToDevice));
    }
    frame_input_context.timestamp =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

    return result;
}

bool IOManager::sendAlert(const AlertMessage & alert) const {
    if (!is_json_sender_ok_ || !send_tcp_enabled_) {
        return false;
    }
    nlohmann::json j(alert);
    bool           success = json_sender_ptr_->send(j);
    if (!success) {
        APP_WARN("Failed to send JSON message: {}", j.dump());
        return false;
    }
    APP_INFO("Send JSON message: {}", j.dump());
    return true;
}
