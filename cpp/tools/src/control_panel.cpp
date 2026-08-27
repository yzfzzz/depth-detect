#include "control_panel.h"

#include "logger_manager.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <utility>

namespace {
int toSlot(float value, float value_min, float scale, int max) {
    if (scale <= 0.0f || !std::isfinite(value)) {
        return 0;
    }
    const int slot = static_cast<int>(std::lround((value - value_min) / scale));
    return std::max(0, std::min(slot, max));
}
}  // namespace

void ControlPanel::onTrackbar(int pos, void * userdata) {
    auto * slider = static_cast<Slider *>(userdata);
    if (slider == nullptr) {
        return;
    }
    slider->pos = pos;
    // 创建期回调（部分后端创建 trackbar 时即回调一次，初始 pos 为 0）直接忽略，
    // 避免把引擎参数误重置为下限
    if (!slider->ready || !slider->apply) {
        return;
    }
    const float value = slider->value_min + static_cast<float>(pos) * slider->scale;
    slider->apply(value);
    APP_INFO("[ControlPanel] {} -> {:.2f}", slider->label, value);
}

ControlPanel::ControlPanel(MotionStateEngine &   motion_engine,
                           const ConfigManager & config,
                           const std::string &   window_name) :
    enabled_(config.isDisplayEnabled() && config.isControlPanelEnabled()),
    window_name_(window_name),
    motion_engine_(motion_engine) {
    if (!enabled_) {
        return;
    }

    // 槽位 0..max，实际值 = value_min + 槽位 * scale
    struct Param {
        const char * label;
        int          max;
        float        value_min;
        float        scale;
        ReadFunc     read;
        ApplyFunc    apply;
    };

    const std::vector<Param> params = {
        { "ema_alpha",                 100, 0.0f, 0.01f, [this]() { return motion_engine_.getEmaAlpha(); },
         [this](float v) {
              motion_engine_.setEmaAlpha(v);
          } },
        { "bbox_jump_ratio_threshold", 100, 0.0f, 0.01f,
         [this]() { return motion_engine_.getBboxJumpRatioThreshold(); },
         [this](float v) {
              motion_engine_.setBboxJumpRatioThreshold(v);
          } },
        { "ttc_warn_threshold",        50,  0.0f, 1.0f,
         [this]() { return motion_engine_.getTtcWarnThreshold(); },
         [this](float v) {
              motion_engine_.setTtcWarnThreshold(v);
          } },
        { "ttc_clear_threshold",       50,  0.0f, 1.0f,
         [this]() { return motion_engine_.getTtcClearThreshold(); },
         [this](float v) {
              motion_engine_.setTtcClearThreshold(v);
          } },
        { "ttc_enter_frames",          10,  1.0f, 1.0f,
         [this]() { return static_cast<float>(motion_engine_.getTtcEnterFrames()); },
         [this](float v) {
              motion_engine_.setTtcEnterFrames(static_cast<int>(std::lround(v)));
          } },
        { "ttc_exit_frames",           10,  1.0f, 1.0f,
         [this]() { return static_cast<float>(motion_engine_.getTtcExitFrames()); },
         [this](float v) {
              motion_engine_.setTtcExitFrames(static_cast<int>(std::lround(v)));
          } },
    };

    for (const auto & p : params) {
        auto slider       = std::make_unique<Slider>();
        slider->label     = p.label;
        slider->max       = p.max;
        slider->value_min = p.value_min;
        slider->scale     = p.scale;
        slider->read      = p.read;
        slider->apply     = p.apply;
        // value 指针为 NULL，滑动条初始位置为 0；稍后由 syncFromEngine 统一对齐
        sliders_.push_back(std::move(slider));
    }

    // 挂载到主显示窗口（由 DisplayManager 创建）；value 指针传 NULL，规避弃用告警
    for (const auto & slider : sliders_) {
        if (cv::createTrackbar(slider->label, window_name_, nullptr, slider->max, onTrackbar,
                               slider.get()) == 0) {
            APP_WARN("[ControlPanel] failed to create trackbar '{}'", slider->label);
        }
    }
    // 创建期回调（如有）已被忽略，这里统一按引擎当前值设置初始槽位；之后再放行回调写回
    syncFromEngine();
    for (auto & slider : sliders_) {
        slider->ready = true;
    }
    APP_INFO("ControlPanel initialized: {} trackbars attached to '{}'", sliders_.size(),
             window_name_);
}

ControlPanel::~ControlPanel() = default;

void ControlPanel::syncFromEngine() {
    for (const auto & slider : sliders_) {
        if (!slider->read) {
            continue;
        }
        const int slot = toSlot(slider->read(), slider->value_min, slider->scale, slider->max);
        if (slot != slider->pos) {
            // setTrackbarPos 不会触发回调，不会与拖动形成"写回 -> 再同步"的反馈环
            cv::setTrackbarPos(slider->label, window_name_, slot);
            slider->pos = slot;
        }
    }
}

void ControlPanel::update() {
    if (!enabled_) {
        return;
    }
    syncFromEngine();
}
