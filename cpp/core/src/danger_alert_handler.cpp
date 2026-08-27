#include "danger_alert_handler.h"

#include "STrack.h"

DangerAlertHandler::DangerAlertHandler(const ConfigManager & config) {
    filter_small_objects_ = config.isFilterSmallObjectsEnabled();
    min_object_area_      = config.getMinObjectArea();
}

bool DangerAlertHandler::isDangerous(const MotionStateInfoRecord & motion) const {
    return motion.ttc_danger;
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

        MotionStateInfoRecord motion = it->second;
        if (!isDangerous(motion)) {
            continue;
        }

        dangerous_objects.push_back(
            { static_cast<int>(track.tlwh_[0]), static_cast<int>(track.tlwh_[1]),
              static_cast<int>(track.tlwh_[2]), static_cast<int>(track.tlwh_[3]), track.class_id_,
              track.track_id_, static_cast<int>(motion.velocity), motion.ttc, true });
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
