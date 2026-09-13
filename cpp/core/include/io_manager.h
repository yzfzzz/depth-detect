#pragma once
#include "config_manager.h"
#include "danger_alert_handler.h"
#include "frame.h"
#include "JsonSender.hpp"

#include <memory.h>
#include <opencv2/core/hal/interface.h>
#include <sys/stat.h>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>

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
              bool        simulate_delay   = false);

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
    double           frame_interval_ms_;      // 模拟节奏的墙钟 tick（毫秒）= 1000/simulate_fps
    double           video_frame_ms_ = 0.0;   // 视频自身帧间隔；0 = 无有效 fps 元数据
    double           skip_accumulator_ = 0.0; // 已产出但尚未消费的帧数（带小数，跨 tick 累计防漂移）
    int              simulate_fps_ = 0;       // 模拟的现场帧率；0 = 跟随视频自身 fps
    bool             simulate_delay_ = false; // 是否启用实时节奏模拟（决定写盘 fps 的取值）
    std::chrono::steady_clock::time_point last_tick_time_; // 上个 tick 唤醒时刻（产出计费窗口起点）
    bool                                  is_first_frame_;   // 标记第一帧

    std::string     video_save_path_;          // 结果视频保存路径（Init 计算，首帧保存时懒初始化写盘）
    double          writer_fps_ = 30.0;        // 写盘 fps = 实际产出帧率（模拟节奏下为 simulate_fps）

    std::unique_ptr<JsonSender> json_sender_ptr_ = nullptr;  // 用于发送 JSON 数据的对象
    std::string                 send_tcp_ip_;                // 发送 JSON 数据的 TCP 地址
    int                         send_tcp_port_;              // 发送 JSON 数据的 TCP 端口
    bool                        send_tcp_enabled_;           // 是否启用 TCP 发送
    // 仅在外接 usb相机时生效
    int                         camera_width_  = 1280;   // 相机采集宽度
    int                         camera_height_ = 720;    // 相机采集高度
    int                         camera_fps_    = 30;     // 相机采集帧率
    bool is_json_sender_ok_                    = false;  // 标记 JsonSender 是否初始化成功
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
