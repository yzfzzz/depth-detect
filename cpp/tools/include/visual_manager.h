#pragma once

#include "config_manager.h"
#include "motion_state_engine.h"
#include "public.h"
#include "STrack.h"

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 前向声明
class DisplayManager;

enum Key_Input {
    ESC   = 27,
    SPACE = 32,
    LEFT  = 65361,
    UP    = 65362,
    RIGHT = 65363,
    DOWN  = 65364,
    A     = 97,
    D     = 100,
    S     = 115,
    W     = 119,
    Q     = 113,
    E     = 101,
    R     = 114,
    Z     = 122,
};

// 显示管理器类：负责窗口管理、显示、鼠标点击等事件
class DisplayManager {
  private:
    bool                enabled_;
    std::vector<STrack> tracks_;
    cv::Mat             depth_map_;
    std::string         window_name_;

    // 在 bounding box 内采样多个点并计算深度均值（鲁棒性更好）
    float computeMeanDepth(const std::vector<float> & tlwh) const;

    // 打印目标详细信息
    void printTargetInfo(const STrack & track) const;

    // 处理鼠标点击
    void handleMouseClick(int x, int y);

    cv::Size display_size_;

  public:
    DisplayManager(const ConfigManager & config,
                   const std::string &   window_name  = "out_frame",
                   cv::Size              display_size = cv::Size(1280, 720 * 2));
    ~DisplayManager();

    // 更新跟踪数据供鼠标回调使用
    void updateData(const std::vector<STrack> & tracks, const cv::Mat & depth_map);

    // 显示图像
    void show(const cv::Mat & frame);

    // 处理键盘事件（空格暂停）
    int handleKey(int key);
    // 等待按键（用于刷新窗口）
    int waitKey(int delay = 1);

    bool isEnabled() const { return enabled_; }

    // 友元函数：鼠标回调
    friend void onMouse(int event, int x, int y, int flags, void * userdata);
};

class DrawingManager {
  public:
    // 传入追踪器引用（或者颜色列表）以及类别名称列表，以便画图时获取颜色和名字
    DrawingManager(const std::vector<std::string> & class_names);

    // 核心绘制函数，画框、文字、以及特殊状态的红叉
    void drawTrackedObject(cv::Mat &                     img,
                           const STrack &                track,
                           const MotionStateInfoRecord & motion_state,
                           cv::Scalar                    color_to_use);

    // 绘制全局信息（FPS、帧数等）
    void drawGlobalInfo(cv::Mat & img, int num_frames, int show_fps, size_t num_tracks);

    // 将检测原图和深度图（彩色可视化）上下拼接为一个新的输出图
    // 把合并图片的代码收归一处
    cv::Mat concatenateFrames(const cv::Mat & rgb_img, const cv::Mat & depth_vis);

  private:
    std::vector<std::string> vClassNames_;
};
