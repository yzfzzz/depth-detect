#pragma once
#include "TcpHandler.hpp"
#include "config_manager.h"
#include "danger_alert_handler.h"
#include "frame.h"

#include <memory.h>
#include <opencv2/core/hal/interface.h>
#include <sys/stat.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

class IOManager {
  public:
    // 构造函数，传入配置管理器以获取保存模式和保存路径，同时传入视频参数
    IOManager(const ConfigManager & config_manager);
    IOManager(std::string save_mode,
              std::string out_dir          = "out_dir",
              std::string send_tcp_ip      = "127.0.0.1",
              int         send_tcp_port    = 12345,
              bool        send_tcp_enabled = false,
              int         camera_width     = 1280,
              int         camera_height    = 720,
              int         camera_fps       = 30,
              int         simulate_fps     = 0,
              bool        simulate_delay   = false,
              double      save_buffer_gb   = 0.5,
              bool        tcp_reconnect    = true,
              int         tcp_check_interval_s  = 5,
              int         tcp_connect_timeout_s = 3);

    FrameMeta Init(const std::string & video_path);

    // 析构函数负责释放系统资源（如关闭写入器）
    ~IOManager();

    // 在每一帧处理完毕后调用此函数，根据 saveMode 自动处理图片保存和/或视频写入
    void saveFrame(const cv::Mat & frame, int num_frames);

    // 判断文件夹是否存在
    static bool dirExists(const std::string & path);

    // 递归创建文件夹
    static void makeDir(const std::string & path);

    // 视频读取和延迟模拟
    // 打开视频源（支持视频文件或相机）
    bool openVideoSource(const std::string & video_path);

    // 关闭视频源
    void closeVideoSource();

    // 读取下一帧。simulate_delay 为 true 时按 simulate_fps 节拍取流：
    bool readNextFrame(FrameInputContext & frame_input_context, bool simulate_delay = false);

    // 获取视频信息
    FrameMeta getVideoFrameMeta() const;
    // 发送告警信息
    bool      sendAlert(const AlertMessage & alert) const;

  private:
    std::string     save_mode_;
    std::string     out_dir_;
    cv::VideoWriter video_writer_;

    cv::VideoCapture video_capture_;
    double frame_interval_ms_;     // 模拟节奏的墙钟 tick（毫秒）= 1000/simulate_fps
    double video_frame_ms_ = 0.0;  // 视频自身帧间隔；0 = 无有效 fps 元数据
    double skip_accumulator_ = 0.0;  // 已产出但尚未消费的帧数（带小数，跨 tick 累计防漂移）
    int  simulate_fps_   = 0;      // 模拟的现场帧率；0 = 跟随视频自身 fps
    bool simulate_delay_ = false;  // 是否启用实时节奏模拟（决定写盘 fps 的取值）
    std::chrono::steady_clock::time_point
         last_tick_time_;          // 上个 tick 唤醒时刻（产出计费窗口起点）
    bool is_first_frame_;          // 标记第一帧

    std::string video_save_path_;  // 结果视频保存路径（Init 计算，首帧保存时懒初始化写盘）
    double writer_fps_ = 30.0;  // 写盘 fps = 实际产出帧率（模拟节奏下为 simulate_fps）

    std::unique_ptr<TcpHandler> tcp_handler_ = nullptr;
    std::string                 send_tcp_ip_;                // 发送 JSON 数据的 TCP 地址
    int                         send_tcp_port_;              // 发送 JSON 数据的 TCP 端口
    bool                        send_tcp_enabled_;           // 是否启用 TCP 发送
    int                         tcp_check_interval_s_  = 5;  // 看门狗探活/重试周期
    int                         tcp_connect_timeout_s_ = 3;  // 单次非阻塞 connect 上限
    bool                        tcp_reconnect_         = true;  // 断联自动重连开关
    // 仅在外接 usb相机时生效
    int                         camera_width_  = 1280;   // 相机采集宽度
    int                         camera_height_ = 720;    // 相机采集高度
    int                         camera_fps_    = 30;     // 相机采集帧率

    // 生产者（主循环）：saveFrame 内 imencode 成 JPG 字节流后非阻塞入队；
    // 消费者（低优先级线程）：CPU 空闲时逐条写盘。缓冲按字节计账，上限
    // save_buffer_limit_ = save_buffer_gb * 1024^3，写满时丢新帧并计数告警。
    struct SaveTask {
        bool is_video = false;     // true=视频帧（顺序写 VideoWriter），false=JPG 图片
        std::string        path;   // 图片保存路径（视频帧忽略）
        cv::Mat            frame;  // 视频模式：原始帧（Mat 引用计数，无深拷贝）
        std::vector<uchar> encoded;    // 图片模式：JPG 编码字节流
        size_t             bytes = 0;  // 本任务占用缓冲的字节数（计账用）
    };

    void startSaveWorker();  // 首次 saveFrame 时拉起消费者线程（幂等）
    void saveWorkerLoop();   // 消费者主循环：排队空后随 stop 标志退出
    void stopSaveWorker();  // 置停止标志、唤醒、排空队列、join（析构时调用，保证不丢数据）
    void enqueueTask(SaveTask && task);  // 非阻塞入队；写满丢弃新帧并计数告警

    std::deque<SaveTask>    save_queue_;
    std::mutex              save_mutex_;
    std::condition_variable save_cv_;
    size_t                  save_buffer_bytes_ = 0;  // 当前占用（字节，仅持锁访问）
    size_t                  save_buffer_limit_ = 0;  // 上限（字节）
    size_t                  save_peak_bytes_   = 0;  // 峰值占用（统计，仅持锁访问）
    std::atomic<size_t>     save_written_{ 0 };      // 已落盘帧数（消费者累计）
    std::atomic<size_t> save_dropped_{ 0 };  // 因缓冲满被丢弃的帧数（生产者累计）
    std::atomic<size_t> save_failed_{ 0 };   // 写盘失败次数（磁盘满/IO 错误）
    std::atomic<bool>   save_worker_started_{ false };
    std::atomic<bool> save_stop_{ false };  // 停止标志（生产者检查、消费者退出条件）
    std::thread save_worker_;
};

struct SendObjectData {
    int  x;  // 左上角横坐标
    int  y;  // 左上角纵坐标
    int  w;  // 宽度
    int  h;  // 高度
    int  class_id_;
    int  track_id_;
    int  vec;
    bool is_danger;  // 是否危险
};

struct SendMessage {
    double                      timestamp;
    int                         frame_id;
    int                         img_w;
    int                         img_h;
    std::vector<SendObjectData> object_data_array;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SendObjectData, x, y, w, h, class_id_, track_id_, vec, is_danger)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SendMessage,
                                   timestamp,
                                   frame_id,
                                   img_w,
                                   img_h,
                                   object_data_array)
