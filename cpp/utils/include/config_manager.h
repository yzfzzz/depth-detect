#pragma once
#include <yaml-cpp/yaml.h>

#include <map>
#include <string>

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
    float                              getMotionVelocityThreshold() const;
    float                              getMotionAccelerationThreshold() const;
    float                              getMotionVelocityHysteresis() const;
    float                              getMotionAccelerationHysteresis() const;
    float                              getYoloNmsThresh() const;
    float                              getYoloConfThresh() const;
    float                              getKfProcessNoiseCov() const;
    float                              getKfMeasurementNoiseCov() const;
    float                              getTtcWarnThreshold() const;
    float                              getTtcClearThreshold() const;
    int                                getTtcEnterFrames() const;
    int                                getTtcExitFrames() const;
    float                              getMinScaleForTtc() const;
    float                              getMinVelocityForTtc() const;
    float                              getEmaAlpha() const;
    float                              getBboxJumpRatioThreshold() const;
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

inline float ConfigManager::getMotionVelocityThreshold() const {
    return config_["motion_state_engine"]["velocity_threshold"].as<float>(5.0f);
}

inline float ConfigManager::getMotionAccelerationThreshold() const {
    return config_["motion_state_engine"]["acceleration_threshold"].as<float>(1.5f);
}

inline float ConfigManager::getMotionVelocityHysteresis() const {
    return config_["motion_state_engine"]["velocity_hysteresis"].as<float>(2.0f);
}

inline float ConfigManager::getMotionAccelerationHysteresis() const {
    return config_["motion_state_engine"]["acceleration_hysteresis"].as<float>(1.0f);
}

inline float ConfigManager::getYoloNmsThresh() const {
    return config_["yolo"]["yolo_nms_thresh"].as<float>(0.4f);
}

inline float ConfigManager::getYoloConfThresh() const {
    return config_["yolo"]["yolo_conf_thresh"].as<float>(0.25f);
}

inline float ConfigManager::getKfProcessNoiseCov() const {
    return config_["motion_state_engine"]["kf_process_noise_cov"].as<float>(2e-2f);
}

inline float ConfigManager::getKfMeasurementNoiseCov() const {
    return config_["motion_state_engine"]["kf_measurement_noise_cov"].as<float>(5e-2f);
}

inline float ConfigManager::getTtcWarnThreshold() const {
    return config_["danger_alert"]["ttc_warn_threshold"].as<float>(3.0f);
}

inline float ConfigManager::getTtcClearThreshold() const {
    return config_["danger_alert"]["ttc_clear_threshold"].as<float>(4.0f);
}

inline int ConfigManager::getTtcEnterFrames() const {
    return config_["danger_alert"]["ttc_enter_frames"].as<int>(3);
}

inline int ConfigManager::getTtcExitFrames() const {
    return config_["danger_alert"]["ttc_exit_frames"].as<int>(10);
}

inline float ConfigManager::getMinScaleForTtc() const {
    return config_["motion_state_engine"]["min_scale_for_ttc"].as<float>(20.0f);
}

inline float ConfigManager::getMinVelocityForTtc() const {
    return config_["motion_state_engine"]["min_velocity_for_ttc"].as<float>(1.0f);
}

inline float ConfigManager::getEmaAlpha() const {
    return config_["motion_state_engine"]["ema_alpha"].as<float>(0.3f);
}

inline float ConfigManager::getBboxJumpRatioThreshold() const {
    return config_["motion_state_engine"]["bbox_jump_ratio_threshold"].as<float>(0.35f);
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
