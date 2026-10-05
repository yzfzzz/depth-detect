#pragma once

#include "config_manager.h"
#include "motion_state_engine.h"

#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 运行时参数控制面板：纯滑动条（OpenCV createTrackbar），挂载在 DisplayManager 的主显示窗口上。
// 注意：主窗口必须先于 ControlPanel 构造（main 中 DisplayManager 先行创建窗口）。
// 面板无编辑状态，update() 返回 void，主循环无需再为面板转发按键。
class ControlPanel {
  public:
    using ReadFunc  = std::function<float()>;      // 读引擎当前生效值（每帧同步用）
    using ApplyFunc = std::function<void(float)>;  // 写回引擎 setter（拖动实时生效）

    // window_name: 挂载滑动条的主显示窗口名（即 DisplayManager 的窗口名）
    ControlPanel(MotionStateEngine &   motion_engine,
                 const ConfigManager & config,
                 const std::string &   window_name);
    ~ControlPanel();

    bool isEnabled() const { return enabled_; }

    // 每帧刷新：把引擎当前生效值同步到各滑动条位置
    void update();

  private:
    struct Slider {
        std::string label;  // 滑动条名（createTrackbar / setTrackbarPos 使用）
        int pos = 0;  // 当前槽位（0 基）；value 指针为 NULL，由回调与 syncFromEngine 维护
        int max = 0;  // 槽位上界（即 createTrackbar 的 count，OpenCV 滑动条为 0 基）
        float value_min = 0.0f;  // 槽位 0 对应的实际浮点值（滑动条下限）
        float scale     = 1.0f;  // 每个槽位对应的实际值增量
        bool ready = false;  // 滑动条创建完成且初始槽位同步后才允许回调写回引擎
        ReadFunc  read;      // 读取引擎当前生效值（每帧同步用）
        ApplyFunc apply;  // 槽位换算后的实际值写回引擎（拖动实时生效）
    };

    // createTrackbar 回调：userdata 指向地址稳定的 Slider
    static void onTrackbar(int pos, void * userdata);

    // 把引擎当前生效值换算为槽位并同步到各滑动条
    void syncFromEngine();

    bool        enabled_;
    std::string window_name_;  // 挂载滑动条的主窗口名（DisplayManager 所有）
    std::vector<std::unique_ptr<Slider>> sliders_;
    MotionStateEngine &                  motion_engine_;
};
