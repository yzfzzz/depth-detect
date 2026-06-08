#pragma once
#include <yaml-cpp/yaml.h>

#include <map>
#include <memory>
#include <string>

// 框架读取配置文件类 - 单例模式
class ConfigManager {
  public:
    static ConfigManager & getInstance(const std::string & config_path = "config.yaml") {
        static ConfigManager instance(config_path);
        return instance;
    }

    ConfigManager(const ConfigManager &)             = delete;
    ConfigManager & operator=(const ConfigManager &) = delete;

    std::map<std::string, std::string> getYoloModelPath() const;
    std::map<std::string, std::string> getDepthModelPath() const;
    int                                getDepthInterval() const;
    std::string                        getSaveMode() const;
    std::string                        getOutDir() const;
    bool                               isDisplayEnabled() const;
    bool                               isSaveEnabled() const;
    float                              getMotionVelocityThreshold() const;
    float                              getMotionAccelerationThreshold() const;
    float                              getYoloNmsThresh() const;
    float                              getYoloConfThresh() const;
    float                              getKfProcessNoiseCov() const;
    float                              getKfMeasurementNoiseCov() const;
    bool                               isUseGPU() const;
    bool                               isOverlapEnabled() const;
    bool                               isLogFileSaveEnabled() const;
    bool                               isLogConsoleOutputEnabled() const;
    std::string                        getLogLevel() const;

  private:
    explicit ConfigManager(const std::string & config_path);

    YAML::Node config_;
};

// Implementation
inline ConfigManager::ConfigManager(const std::string & config_path) {
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

inline std::string ConfigManager::getSaveMode() const {
    return config_["io_manager"]["save_mode"].as<std::string>("none");
}

inline std::string ConfigManager::getOutDir() const {
    return config_["io_manager"]["out_dir"].as<std::string>("out_dir");
}

inline bool ConfigManager::isDisplayEnabled() const {
    return config_["display_manager"]["is_display"].as<bool>(false);
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

inline bool ConfigManager::isOverlapEnabled() const {
    return config_["prefer"]["overlap"].as<bool>(true);
}
