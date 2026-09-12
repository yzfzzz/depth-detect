#include "danger_alert_handler.h"

#include "STrack.h"

#include <cmath>

 // namespace

DangerAlertHandler::DangerAlertHandler(const ConfigManager & config) {
    filter_small_objects_ = config.isFilterSmallObjectsEnabled();
    min_object_area_      = config.getMinObjectArea();
}

bool DangerAlertHandler::isDangerous(const MotionStateInfoRecord & motion) const {
    // 危险与否只看“快速靠近”判定（approach 关闭时该字段恒为 false）
    return motion.approach_alarm;
}

AlertMessage DangerAlertHandler::buildAlert(const FrameInputContext &  frame_input,
                                            const InferOutputContext & infer_output) const {
    std::vector<AlertObjectData> dangerous_objects;

    for (const auto & track : infer_output.tracked_objects) {
        // 过滤小目标
        if (filter_small_objects_) {
            float s = track.tlwh[2] * track.tlwh[3];
            if(track.class_id == 2 && s <= min_object_area_ * 4){
                continue;
            }
            else if(s <= min_object_area_){
                continue;
            }
        }

        // 查找本帧的判定结果（不在接近单元内的目标没有记录）
        auto it = infer_output.motion_records.find(track.track_id);
        if (it == infer_output.motion_records.end()) {
            continue;
        }

        const MotionStateInfoRecord & motion = it->second;
        int class_id = track.class_id;
        int d = static_cast<int>(std::lround(track.distance_));


        // 全部发送（depth 四舍五入到整数米）
        dangerous_objects.push_back(
            { static_cast<int>(track.tlwh[0]), static_cast<int>(track.tlwh[1]),
              static_cast<int>(track.tlwh[2]), static_cast<int>(track.tlwh[3]),
              d, class_id, track.track_id, 0, -1.0f,
              isDangerous(motion) });
    }

    if (dangerous_objects.empty()) {
        return AlertMessage{ frame_input.timestamp,
                             frame_input.frame_id,
                             frame_input.meta.img_w,
                             frame_input.meta.img_h,
                             {} };
    }

    return AlertMessage{ frame_input.timestamp, frame_input.frame_id, frame_input.meta.img_w,
                         frame_input.meta.img_h, std::move(dangerous_objects) };
}
