#include "io_manager.h"

#include "frame.h"
#include "logger_manager.h"
#include "public.h"

#include <cstdlib>  // For system()

IOManager::IOManager(const ConfigManager & config_manager) :
    IOManager(config_manager.getSaveMode(),
              config_manager.getOutDir(),
              config_manager.getSendTcpIp(),
              config_manager.getSendTcpPort(),
              config_manager.isSendTcpEnabled()) {}

IOManager::IOManager(std::string save_mode,
                     std::string out_dir,
                     std::string send_tcp_ip,
                     int         send_tcp_port,
                     bool        send_tcp_enabled) :
    save_mode_(std::move(save_mode)),
    out_dir_(std::move(out_dir)),
    send_tcp_ip_(std::move(send_tcp_ip)),
    send_tcp_port_(send_tcp_port),
    send_tcp_enabled_(send_tcp_enabled) {}

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
    // TODO: 异步读取
    if (!video_capture_.isOpened()) {
        return false;
    }

    // 帧延迟模拟：若上一帧处理耗时超过帧间隔，跳过多余帧以追赶实时播放进度
    // 避免视频播放与实际处理速度脱节导致的帧积压
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
    // 读取当前帧并同步拷贝到 GPU，供 CUDA 预处理使用
    bool        result       = video_capture_.read(frame_input_context.raw_img);
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
    // 更新下一帧的处理开始时间
    last_frame_start_time_ = std::chrono::steady_clock::now();
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
