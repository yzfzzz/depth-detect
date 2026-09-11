#pragma once
#include "config_manager.h"
#include "frame.h"

#include <nlohmann/json.hpp>
#include <vector>

struct AlertObjectData {
    int x;
    int y;
    int w;
    int h;
    int depth;
    int class_id;
    int track_id;
    int velocity;  // 旧 TTC 路遗留字段：现已不做 TTC 估计，固定为 0（保持 JSON 兼容）
    float ttc;  // 同上：固定为 -1（无效）
    bool  is_danger;
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
                                   depth,
                                   class_id,
                                   track_id,
                                   velocity,
                                   ttc,
                                   is_danger)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AlertMessage, timestamp, frame_id, img_w, img_h, objects)

// 危险报警处理器：危险判定（“快速靠近”approach）已由 MotionStateEngine 完成
// （见 MotionStateInfoRecord::approach_alarm 与 approach_detector.h），本类只负责
// 小目标过滤与报警消息组装：快速靠近的目标即 is_danger=true 上报。
class DangerAlertHandler {
  public:
    explicit DangerAlertHandler(const ConfigManager & config);

    AlertMessage buildAlert(const FrameInputContext &  frame_input,
                            const InferOutputContext & infer_output) const;

  private:
    bool isDangerous(const MotionStateInfoRecord & motion) const;

    bool  filter_small_objects_;  // 是否过滤小目标
    float min_object_area_;       // 最小目标面积阈值
};
