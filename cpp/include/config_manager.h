#pragma once
#include <yaml-cpp/yaml.h>

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

    std::string getYoloEnginePath() const;
    std::string getDepthEnginePath() const;
    int         getDepthInterval() const;
    std::string getSaveMode() const;
    std::string getOutDir() const;
    bool        isDisplayEnabled() const;
    bool        isSaveEnabled() const;
    float       getMotionVelocityThreshold() const;
    float       getMotionAccelerationThreshold() const;
    float       getYoloNmsThresh() const;
    float       getYoloConfThresh() const;
    float       getKfProcessNoiseCov() const;
    float       getKfMeasurementNoiseCov() const;

  private:
    explicit ConfigManager(const std::string & config_path);

    YAML::Node  config_;
    std::string yolo_trt_file_;
    std::string depth_trt_file_;
    std::string save_mode_;
    std::string out_dir_;
    int         depth_interval_;
    bool        is_display_;
    bool        is_save_;

    float motion_velocity_threshold_;
    float motion_acceleration_threshold_;

    float yolo_nms_thresh_;
    float yolo_conf_thresh_;
    float kf_process_noise_cov_;
    float kf_measurement_noise_cov_;

    // Implementation
    void initialize();
};

// Implementation
inline ConfigManager::ConfigManager(const std::string & config_path) {
    config_     = YAML::LoadFile(config_path);
    is_display_ = config_["display_manager"]["is_display"].as<bool>(false);

    yolo_trt_file_    = config_["yolo"]["yolo_engine"].as<std::string>();
    yolo_nms_thresh_  = config_["yolo"]["yolo_nms_thresh"].as<float>(0.4f);
    yolo_conf_thresh_ = config_["yolo"]["yolo_conf_thresh"].as<float>(0.25f);

    depth_trt_file_ = config_["depth"]["depth_engine"].as<std::string>();
    depth_interval_ = config_["depth"]["depth_interval"].as<int>(1);

    is_save_   = config_["io_manager"]["is_save"].as<bool>(false);
    save_mode_ = config_["io_manager"]["save_mode"].as<std::string>("none");
    out_dir_   = config_["io_manager"]["out_dir"].as<std::string>("out_dir");

    // 运动状态引擎相关配置
    motion_velocity_threshold_ =
        config_["motion_state_engine"]["velocity_threshold"].as<float>(5.0f);
    motion_acceleration_threshold_ =
        config_["motion_state_engine"]["acceleration_threshold"].as<float>(1.5f);
    kf_process_noise_cov_ = config_["motion_state_engine"]["kf_process_noise_cov"].as<float>(2e-2f);
    kf_measurement_noise_cov_ =
        config_["motion_state_engine"]["kf_measurement_noise_cov"].as<float>(5e-2f);
}

inline std::string ConfigManager::getYoloEnginePath() const {
    return yolo_trt_file_;
}

inline std::string ConfigManager::getDepthEnginePath() const {
    return depth_trt_file_;
}

inline int ConfigManager::getDepthInterval() const {
    return depth_interval_;
}

inline std::string ConfigManager::getSaveMode() const {
    return save_mode_;
}

inline std::string ConfigManager::getOutDir() const {
    return out_dir_;
}

inline bool ConfigManager::isDisplayEnabled() const {
    return is_display_;
}

inline bool ConfigManager::isSaveEnabled() const {
    return is_save_;
}

inline float ConfigManager::getMotionVelocityThreshold() const {
    return motion_velocity_threshold_;
}

inline float ConfigManager::getMotionAccelerationThreshold() const {
    return motion_acceleration_threshold_;
}

inline float ConfigManager::getYoloNmsThresh() const {
    return yolo_nms_thresh_;
}

inline float ConfigManager::getYoloConfThresh() const {
    return yolo_conf_thresh_;
}

inline float ConfigManager::getKfProcessNoiseCov() const {
    return kf_process_noise_cov_;
}

inline float ConfigManager::getKfMeasurementNoiseCov() const {
    return kf_measurement_noise_cov_;
}
