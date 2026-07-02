#pragma once
#include "config_manager.h"
#include "frame.h"

#include <nlohmann/json.hpp>
#include <vector>

struct AlertObjectData {
    int  x;
    int  y;
    int  w;
    int  h;
    int  class_id;
    int  track_id;
    int  velocity;
    bool is_danger;
};

struct AlertMessage {
    double                       timestamp;
    int                          frame_id;
    int                          img_w;
    int                          img_h;
    std::vector<AlertObjectData> objects;

    bool has_value() const { return !objects.empty(); }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AlertObjectData,
                                   x,
                                   y,
                                   w,
                                   h,
                                   class_id,
                                   track_id,
                                   velocity,
                                   is_danger)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AlertMessage, timestamp, frame_id, img_w, img_h, objects)

// 危险报警处理器类,负责从运动估计引擎中提取危险目标并生成报警Json消息
class DangerAlertHandler {
  public:
    explicit DangerAlertHandler(const ConfigManager & config);

    // 从运动估计引擎中提取危险目标，生成报警消息
    AlertMessage buildAlert(const FrameInputContext &  frame_input,
                            const InferOutputContext & infer_output) const;

  private:
    // 危险判定规则（可配置、可扩展）
    bool isDangerous(const MotionStateInfoRecord & motion) const;

    // 可配置的危险规则参数
    bool  filter_small_objects_;  // 是否过滤小目标
    float min_object_area_;       // 最小目标面积阈值
    // 未来可扩展：危险状态组合列表、类别白名单等
};
