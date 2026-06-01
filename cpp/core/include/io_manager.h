#pragma once
#include "config_manager.h"
#include "frame.h"

#include <memory.h>
#include <opencv2/core/hal/interface.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <string>

class IOManager {
  public:
    // 构造函数，传入配置管理器以获取保存模式和保存路径，同时传入视频参数
    IOManager(const ConfigManager & config_manager);

    FrameMeta Init(const std::string & video_path);

    // 析构函数负责释放系统资源（如关闭写入器）
    ~IOManager();

    // 在每一帧处理完毕后调用此函数，根据 saveMode 自动处理图片保存和/或视频写入
    void saveFrame(const cv::Mat & frame, int num_frames);

    // 判断文件夹是否存在
    static bool dirExists(const std::string & path);

    // 递归创建文件夹
    static void makeDir(const std::string & path);

    // ===== 视频读取和延迟模拟 =====
    // 打开视频源（支持视频文件或相机）
    bool openVideoSource(const std::string & video_path);

    // 关闭视频源
    void closeVideoSource();

    // 读取下一帧。如果 simulate_delay 为 true，则会根据上一帧处理耗时跳过对应帧数
    bool readNextFrame(FrameInputContext & frame_input_context, bool simulate_delay = false);

    // 获取视频信息
    FrameMeta getVideoFrameMeta() const;

  private:
    std::string     save_mode_;
    std::string     out_dir_;
    cv::VideoWriter video_writer_;

    cv::VideoCapture                      video_capture_;
    double                                frame_interval_ms_;  // 每帧的时间间隔（毫秒）
    std::chrono::steady_clock::time_point last_frame_start_time_;
    bool                                  is_first_frame_;     // 标记第一帧
};
