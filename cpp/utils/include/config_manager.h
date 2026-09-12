#pragma once
#include <yaml-cpp/yaml.h>

#include <map>
#include <string>
#include <vector>

// 框架读取配置文件类 - 单例模式
class ConfigManager {
  public:
    ConfigManager(const std::string config_path);
    std::map<std::string, std::string> getYoloModelPath() const;
    std::map<std::string, std::string> getDepthModelPath() const;
    int                                getDepthInterval() const;
    bool                               isDepthEnabled() const;
    std::string                        getSaveMode() const;
    std::string                        getOutDir() const;
    bool                               isDisplayEnabled() const;
    bool                               isControlPanelEnabled() const;
    bool                               isSaveEnabled() const;
    float                              getYoloNmsThresh() const;
    float                              getYoloConfThresh() const;
    // ---- 快速靠近（approach）检测参数：方案 d 的 C++ 移植，见 approach_detector.h ----
    bool                               isApproachEnabled() const;
    std::string                        getApproachFilterMode() const;
    int                                getApproachWarmup() const;
    float                              getApproachThrDepth() const;
    float                              getApproachThrHeight() const;
    int                                getApproachRecentW() const;
    float                              getApproachScoreThr() const;
    int                                getApproachConfirm() const;
    float                              getApproachExitScoreThr() const;
    int                                getApproachExitConfirm() const;
    bool                               isUseGPU() const;
    bool                               isOverlapEnabled() const;
    bool                               isLogFileSaveEnabled() const;
    bool                               isLogConsoleOutputEnabled() const;
    std::string                        getLogLevel() const;
    void                               setUseGPU(bool use_gpu);
    void                               setLogLevel(const std::string & log_level);
    std::string                        getSendTcpIp() const;
    int                                getSendTcpPort() const;
    bool                               isSendTcpEnabled() const;
    bool                               isTrackLogEnabled() const;
    bool                               isFilterSmallObjectsEnabled() const;
    float                              getMinObjectArea() const;
    int                                getCameraWidth() const;
    int                                getCameraHeight() const;
    int                                getCameraFps() const;
    std::string                        getDepthModelType() const;
    // ---- ByteTrack 跟踪器参数：对应 bytetrack_shaky.yaml，见 bytetrack/BYTETracker.h ----
    int                                getTrackerTrackBuffer() const;
    float                              getTrackHighThresh() const;
    float                              getTrackLowThresh() const;
    float                              getNewTrackThresh() const;
    float                              getMatchThresh() const;

  private:
    YAML::Node config_;
};

// Implementation
inline ConfigManager::ConfigManager(const std::string config_path) {
    config_ = YAML::LoadFile(config_path);
}

inline std::map<std::string, std::string> ConfigManager::getYoloModelPath() const {
    std::map<std::string, std::string> model_paths;
    const auto &                       yolo_model_paths = config_["yolo"]["yolo_model_path"];
    for (const auto & model_path : yolo_model_paths) {
        model_paths[model_path["type"].as<std::string>()] = model_path["path"].as<std::string>();
    }
    return model_paths;
}

inline std::map<std::string, std::string> ConfigManager::getDepthModelPath() const {
    std::map<std::string, std::string> model_paths;
    const auto &                       depth_model_paths = config_["depth"]["depth_model_path"];
    for (const auto & model_path : depth_model_paths) {
        model_paths[model_path["type"].as<std::string>()] = model_path["path"].as<std::string>();
    }
    return model_paths;
}

inline int ConfigManager::getDepthInterval() const {
    return config_["depth"]["depth_interval"].as<int>(1);
}

inline bool ConfigManager::isDepthEnabled() const {
    return config_["depth"]["enabled"].as<bool>(true);
}

inline std::string ConfigManager::getSaveMode() const {
    return config_["io_manager"]["save_mode"].as<std::string>("none");
}

inline std::string ConfigManager::getOutDir() const {
    return config_["io_manager"]["out_dir"].as<std::string>("out_dir");
}

inline bool ConfigManager::isDisplayEnabled() const {
    return config_["display_manager"]["is_display"].as<bool>(false);
}

inline bool ConfigManager::isControlPanelEnabled() const {
    return config_["display_manager"]["enable_control_panel"].as<bool>(true);
}

inline bool ConfigManager::isSaveEnabled() const {
    return config_["io_manager"]["is_save"].as<bool>(false);
}

inline float ConfigManager::getYoloNmsThresh() const {
    return config_["yolo"]["yolo_nms_thresh"].as<float>(0.4f);
}

inline float ConfigManager::getYoloConfThresh() const {
    return config_["yolo"]["yolo_conf_thresh"].as<float>(0.25f);
}

// --------------------------------------------------------------------------- //
// 快速靠近（approach）检测参数
// 默认值 = mini_python/pipeline.py 调好的最优参数：
//   --approach d --filter one_euro --approach-score-thr 0.45 --approach-confirm 3
//   --approach-thr-depth 0.15 --approach-thr-height 0.20
//   --approach-exit-confirm 3 --approach-exit-score-thr 0.2
// 注：尺度判据已由“框面积变化”改为“框高变化”（thr_height）
// --------------------------------------------------------------------------- //

inline bool ConfigManager::isApproachEnabled() const {
    return config_["motion_state_engine"]["approach"]["enabled"].as<bool>(true);
}

inline std::string ConfigManager::getApproachFilterMode() const {
    return config_["motion_state_engine"]["approach"]["filter"].as<std::string>("one_euro");
}

inline int ConfigManager::getApproachWarmup() const {
    return config_["motion_state_engine"]["approach"]["warmup"].as<int>(30);
}

inline float ConfigManager::getApproachThrDepth() const {
    return config_["motion_state_engine"]["approach"]["thr_depth"].as<float>(0.15f);
}

inline float ConfigManager::getApproachThrHeight() const {
    return config_["motion_state_engine"]["approach"]["thr_height"].as<float>(0.30f);
}

inline int ConfigManager::getApproachRecentW() const {
    return config_["motion_state_engine"]["approach"]["recent_w"].as<int>(10);
}

inline float ConfigManager::getApproachScoreThr() const {
    return config_["motion_state_engine"]["approach"]["score_thr"].as<float>(0.45f);
}

inline int ConfigManager::getApproachConfirm() const {
    return config_["motion_state_engine"]["approach"]["confirm"].as<int>(3);
}

inline float ConfigManager::getApproachExitScoreThr() const {
    return config_["motion_state_engine"]["approach"]["exit_score_thr"].as<float>(0.2f);
}

inline int ConfigManager::getApproachExitConfirm() const {
    return config_["motion_state_engine"]["approach"]["exit_confirm"].as<int>(3);
}

inline bool ConfigManager::isLogFileSaveEnabled() const {
    return config_["logger"]["save_file"].as<bool>(true);
}

inline bool ConfigManager::isLogConsoleOutputEnabled() const {
    return config_["logger"]["console_output"].as<bool>(true);
}

inline std::string ConfigManager::getLogLevel() const {
    return config_["logger"]["log_level"].as<std::string>("info");
}

inline bool ConfigManager::isUseGPU() const {
    return config_["prefer"]["use_gpu"].as<bool>(false);
}

inline void ConfigManager::setUseGPU(bool use_gpu) {
    config_["prefer"]["use_gpu"] = use_gpu;
}

inline bool ConfigManager::isOverlapEnabled() const {
    return config_["prefer"]["overlap"].as<bool>(true);
}

inline void ConfigManager::setLogLevel(const std::string & log_level) {
    config_["logger"]["log_level"] = log_level;
}

inline std::string ConfigManager::getSendTcpIp() const {
    return config_["io_manager"]["send_tcp_ip"].as<std::string>("127.0.0.1");
}

inline int ConfigManager::getSendTcpPort() const {
    return config_["io_manager"]["send_tcp_port"].as<int>(12345);
}

inline bool ConfigManager::isSendTcpEnabled() const {
    return config_["io_manager"]["send_tcp"].as<bool>(false);
}

inline bool ConfigManager::isTrackLogEnabled() const {
    return config_["io_manager"]["save_track_log"].as<bool>(false);
}

inline bool ConfigManager::isFilterSmallObjectsEnabled() const {
    return config_["danger_alert"]["is_filter_small_objects"].as<bool>(true);
}

inline float ConfigManager::getMinObjectArea() const {
    return config_["danger_alert"]["min_object_area"].as<float>(20.0f);
}

inline int ConfigManager::getCameraWidth() const {
    return config_["camera"]["width"].as<int>(1280);
}

inline int ConfigManager::getCameraHeight() const {
    return config_["camera"]["height"].as<int>(720);
}

inline int ConfigManager::getCameraFps() const {
    return config_["camera"]["fps"].as<int>(30);
}

inline std::string ConfigManager::getDepthModelType() const {
    return config_["depth"]["model_type"].as<std::string>("lite_mono");
}

// --------------------------------------------------------------------------- //
// ByteTrack 跟踪器参数（yaml 根节点 "bytetrack"，缺省时用 ByteTrack 官方默认值）
// 抖动场景参考调参（bytetrack_shaky.yaml）：
//   track_high_thresh: 0.30  track_low_thresh: 0.05  new_track_thresh: 0.60
//   track_buffer: 120  match_thresh: 0.90
// --------------------------------------------------------------------------- //

inline int ConfigManager::getTrackerTrackBuffer() const {
    return config_["bytetrack"]["track_buffer"].as<int>(90);
}

inline float ConfigManager::getTrackHighThresh() const {
    return config_["bytetrack"]["track_high_thresh"].as<float>(0.5f);
}

inline float ConfigManager::getTrackLowThresh() const {
    return config_["bytetrack"]["track_low_thresh"].as<float>(0.1f);
}

inline float ConfigManager::getNewTrackThresh() const {
    return config_["bytetrack"]["new_track_thresh"].as<float>(0.6f);
}

inline float ConfigManager::getMatchThresh() const {
    return config_["bytetrack"]["match_thresh"].as<float>(0.8f);
}
