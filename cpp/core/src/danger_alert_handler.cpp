#include "danger_alert_handler.h"

#include "STrack.h"

DangerAlertHandler::DangerAlertHandler(const ConfigManager & config) {
    // 从 config 读取危险判定策略
    filter_small_objects_ = config.isFilterSmallObjectsEnabled();
    min_object_area_      = config.getMinObjectArea();
}

bool DangerAlertHandler::isDangerous(const MotionStateInfoRecord & motion) const {
    // 当前规则：正在接近 + 加速中 = 危险
    // 未来可扩展为策略链模式
    return (motion.state_vec == MotionState::APPROACH && motion.state_acc == MotionState::ACCELE);
}

AlertMessage DangerAlertHandler::buildAlert(const FrameInputContext &  frame_input,
                                            const InferOutputContext & infer_output) const {
    std::vector<AlertObjectData> dangerous_objects;

    for (const auto & track : infer_output.tracked_objects) {
        // 过滤小目标
        if (filter_small_objects_ && track.tlwh_[2] * track.tlwh_[3] <= min_object_area_) {
            continue;
        }

        // 查找运动状态
        auto it = infer_output.motion_records.find(track.track_id_);
        if (it == infer_output.motion_records.end()) {
            continue;
        }

        // 危险判定
        if (!isDangerous(it->second)) {
            continue;
        }

        dangerous_objects.push_back(
            { static_cast<int>(track.tlwh_[0]), static_cast<int>(track.tlwh_[1]),
              static_cast<int>(track.tlwh_[2]), static_cast<int>(track.tlwh_[3]), track.class_id_,
              track.track_id_, static_cast<int>(it->second.velocity), true });
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
