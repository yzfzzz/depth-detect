#include "io_manager.h"

#include "frame.h"
#include "logger_manager.h"
#include "public.h"

#include <algorithm>  // std::all_of
#include <cctype>     // std::isdigit
#include <cstdio>
#include <cstdlib>    // For system()
#include <opencv2/imgcodecs.hpp>
#include <sstream>  // std::ostringstream（结果视频文件名的时间戳格式化）
#include <thread>   // std::this_thread::sleep_for（实时节奏模拟）

#ifdef __linux__
#    include <pthread.h>
#    include <sched.h>
#    include <sys/resource.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

namespace {
// GB -> 字节；下限钳到 0.1 GB（约 100 MB），防止配 0 导致缓冲永远为满
size_t gbToBytes(double gb) {
    const double clamped = std::max(0.1, gb);
    return static_cast<size_t>(clamped * 1024.0 * 1024.0 * 1024.0);
}
}  // namespace

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
              config_manager.isSimulateDelayEnabled(),
              config_manager.getSaveBufferGb(),
              config_manager.isTcpReconnectEnabled(),
              config_manager.getTcpCheckIntervalS(),
              config_manager.getTcpConnectTimeoutS()) {}

IOManager::IOManager(std::string save_mode,
                     std::string out_dir,
                     std::string send_tcp_ip,
                     int         send_tcp_port,
                     bool        send_tcp_enabled,
                     int         camera_width,
                     int         camera_height,
                     int         camera_fps,
                     int         simulate_fps,
                     bool        simulate_delay,
                     double      save_buffer_gb,
                     bool        tcp_reconnect,
                     int         tcp_check_interval_s,
                     int         tcp_connect_timeout_s) :
    save_mode_(std::move(save_mode)),
    out_dir_(std::move(out_dir)),
    send_tcp_ip_(std::move(send_tcp_ip)),
    send_tcp_port_(send_tcp_port),
    send_tcp_enabled_(send_tcp_enabled),
    camera_width_(camera_width),
    camera_height_(camera_height),
    camera_fps_(camera_fps),
    simulate_fps_(simulate_fps),
    simulate_delay_(simulate_delay),
    save_buffer_limit_(gbToBytes(save_buffer_gb)),
    tcp_reconnect_(tcp_reconnect),
    tcp_check_interval_s_(tcp_check_interval_s),
    tcp_connect_timeout_s_(tcp_connect_timeout_s) {}

FrameMeta IOManager::Init(const std::string & video_path) {
    // 如果需要保存图片，检查目标文件夹并创建
    if (save_mode_ != "none" && out_dir_ != "" && !dirExists(out_dir_)) {
        makeDir(out_dir_);
    }
    if (save_mode_ != "none") {
        // 注意：缓冲按原始帧字节计账（消费者侧才编码），容量与帧分辨率挂钩
        APP_INFO("[SaveWorker] async saving enabled: mode={}, buffer limit={:.1f} MB "
                 "(accounted by raw frame bytes, e.g. 1280x720 BGR ~2.6 MB/frame)",
                 save_mode_, static_cast<double>(save_buffer_limit_) / (1024.0 * 1024.0));
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
        // 网络能力归 third_party/JsonSenderTest 的 TcpHandler：连接建立与断联重连全在
        // 它的看门狗线程里（首连非阻塞，服务端未启动不会被卡死，断开后按
        // tcp_check_interval_s 周期重连）。
        TcpHandler::setLogSink([](TcpHandler::LogLevel level, const std::string & msg) {
            switch (level) {
                case TcpHandler::LogLevel::WARN:
                    APP_WARN("{}", msg);
                    break;
                case TcpHandler::LogLevel::ERROR:
                    APP_ERROR("{}", msg);
                    break;
                default:
                    APP_INFO("{}", msg);
                    break;
            }
        });
        tcp_handler_ =
            std::make_unique<TcpHandler>(send_tcp_ip_, send_tcp_port_, tcp_check_interval_s_,
                                         tcp_connect_timeout_s_, tcp_reconnect_);
        tcp_handler_->start();
    } else {
        APP_WARN("TCP sending is disabled.");
    }
    return frame_meta;
}

IOManager::~IOManager() {
    // 先停 TCP 看门狗（放弃未完成的重连、打连接统计），再停落盘线程——
    // 消费者还在用 VideoWriter 写帧；排空保证退出时缓冲内未落盘数据完整写出
    if (tcp_handler_) {
        tcp_handler_->stop();
    }
    stopSaveWorker();
    if (video_writer_.isOpened()) {
        video_writer_.release();
    }
    closeVideoSource();
}

void IOManager::saveFrame(const cv::Mat & frame, int num_frames) {
    const bool save_image = (save_mode_ == "images" || save_mode_ == "both");
    const bool save_video = (save_mode_ == "video" || save_mode_ == "both");
    if (!save_image && !save_video) {
        return;
    }

    // 首次保存时拉起消费者线程（幂等，仅执行一次）
    if (!save_worker_started_.load()) {
        startSaveWorker();
    }
    if (save_stop_) {
        return;  // 已进入退出流程，拒绝新任务
    }


    const size_t frame_bytes = static_cast<size_t>(frame.total()) * frame.elemSize();

    if (save_image) {
        SaveTask task;
        task.path  = out_dir_ + "/frame_" + std::to_string(num_frames) + ".jpg";
        task.frame = frame;
        task.bytes = frame_bytes;
        enqueueTask(std::move(task));
    }

    if (save_video) {
        // 首帧懒初始化（生产者侧执行；之后只有消费者调用 write，无并发访问）：
        // 写盘尺寸取实际落盘帧尺寸（深度启用时为上下拼接的 2 倍高）
        if (!video_writer_.isOpened() && !video_save_path_.empty()) {
            video_writer_.open(video_save_path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               writer_fps_, frame.size());
            if (!video_writer_.isOpened()) {
                APP_ERROR("Failed to initialize VideoWriter at {}", video_save_path_);
                video_save_path_.clear();  // 防止之后每帧重复尝试与报错
            }
        }
        if (video_writer_.isOpened()) {
            SaveTask task;
            task.is_video = true;
            task.frame    = frame;  // 与图片任务共享同一块缓冲（引用计数，无额外内存）
            task.bytes    = frame_bytes;
            enqueueTask(std::move(task));
        }
    }
}

void IOManager::startSaveWorker() {
    bool expected = false;
    if (!save_worker_started_.compare_exchange_strong(expected, true)) {
        return;  // 已启动（或并发启动中由赢家执行）
    }
    save_worker_ = std::thread(&IOManager::saveWorkerLoop, this);
}

void IOManager::saveWorkerLoop() {
#ifdef __linux__
    // Linux "CPU 空闲"调度：线程内自降优先级，无需特权（不需要 CAP_SYS_NICE）。
    // 优先 SCHED_IDLE（低于所有普通线程，绝对让路）→ 失败退 nice 19 → 再失败保持默认
    sched_param sp;
    sp.sched_priority = 0;
    if (pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp) == 0) {
        APP_INFO("[SaveWorker] scheduling: SCHED_IDLE (runs only when CPU idle)");
    } else if (setpriority(PRIO_PROCESS, static_cast<int>(syscall(SYS_gettid)), 19) == 0) {
        APP_INFO("[SaveWorker] scheduling: nice 19 (SCHED_IDLE unavailable)");
    } else {
        APP_WARN("[SaveWorker] priority lowering failed, running at default niceness");
    }
#else
    APP_WARN("[SaveWorker] non-Linux platform: idle-priority scheduling not applied");
#endif

    for (;;) {
        SaveTask task;
        {
            std::unique_lock<std::mutex> lock(save_mutex_);
            // 队列空则等待；stop 置位后继续排空剩余任务（drain），保证退出不丢数据
            save_cv_.wait(lock, [this] { return save_stop_ || !save_queue_.empty(); });
            if (save_queue_.empty()) {
                break;  // stop 且已排空
            }
            task = std::move(save_queue_.front());
            save_queue_.pop_front();
            save_buffer_bytes_ -= task.bytes;
        }

        // 编码 + 写盘在锁外执行：JPEG 压缩与 fwrite 耗时不占用生产者的入队路径
        bool ok = true;
        if (task.is_video) {
            if (video_writer_.isOpened()) {
                video_writer_.write(task.frame);
            }
        } else {
            static const std::vector<int> kEncodeParams{ cv::IMWRITE_JPEG_QUALITY, 75 };
            ok = cv::imwrite(task.path, task.frame, kEncodeParams);
        }
        save_written_ += 1;
        if (!ok) {
            save_failed_ += 1;
            const size_t fail_total = save_failed_.load();
            if (fail_total == 1 || fail_total % 100 == 0) {
                APP_WARN("[SaveWorker] disk write failed (total {}): {}", fail_total, task.path);
            }
        }
    }

    APP_INFO("[SaveWorker] exit: written={}, dropped={}, io_failed={}, peak_buffer={:.1f} MB",
             save_written_.load(), save_dropped_.load(), save_failed_.load(),
             static_cast<double>(save_peak_bytes_) / (1024.0 * 1024.0));
}

void IOManager::stopSaveWorker() {
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        if (!save_worker_started_.load() || save_stop_) {
            return;
        }
        save_stop_ = true;
    }
    save_cv_.notify_all();
    if (save_worker_.joinable()) {
        save_worker_.join();
    }
}

void IOManager::enqueueTask(SaveTask && task) {
    bool   dropped  = false;
    size_t drop_seq = 0;
    double used_mb  = 0.0;
    double limit_mb = 0.0;
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        // 满策略：丢新帧 + 计数告警（不阻塞生产者，保实时节拍）。
        // 阻塞生产者会让视频时间与墙钟的 1:1 同步漂移，故不采用
        if (save_buffer_bytes_ + task.bytes > save_buffer_limit_) {
            save_dropped_ += 1;
            dropped  = true;
            drop_seq = save_dropped_.load();
            used_mb  = static_cast<double>(save_buffer_bytes_) / (1024.0 * 1024.0);
            limit_mb = static_cast<double>(save_buffer_limit_) / (1024.0 * 1024.0);
        } else {
            save_buffer_bytes_ += task.bytes;
            save_peak_bytes_ = std::max(save_peak_bytes_, save_buffer_bytes_);
            save_queue_.push_back(std::move(task));
        }
    }
    // 告警在锁外打：日志写文件本身也有 IO 开销，不能拖住消费者出队
    if (dropped) {
        if (drop_seq == 1 || drop_seq % 100 == 0) {
            APP_WARN(
                "[SaveWorker] buffer full ({:.1f} / {:.1f} MB), dropped {} frame(s) total "
                "(disk too slow; raise io_manager.save_buffer_gb or check storage)",
                used_mb, limit_mb, drop_seq);
        }
        return;
    }
    save_cv_.notify_one();
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
    if (!send_tcp_enabled_ || !tcp_handler_) {
        return false;
    }
    nlohmann::json j(alert);
    // TcpHandler 状态感知发送：非 CONNECTED 直接丢弃并计数（断联期间告警不重发）；
    // 发送失败时由 JsonSenderTcp 内部标记 BROKEN 并唤醒看门狗重连
    bool           success = tcp_handler_->sendJson(j);
    if (!success) {
        APP_WARN("Failed to send JSON message: {}", j.dump());
        return false;
    }
    APP_INFO("Send JSON message: {}", j.dump());
    return true;
}
