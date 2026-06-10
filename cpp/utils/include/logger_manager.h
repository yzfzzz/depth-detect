#pragma once
#include "config_manager.h"

#include <NvInfer.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>

/**
 * @class Logger
 * @brief TensorRT 日志适配器 - 将 TensorRT 日志写入 spdlog
 */
class Logger : public nvinfer1::ILogger {
  public:
    nvinfer1::ILogger::Severity reportable_severity_;

    Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
        reportable_severity_(severity) {}

    void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override {
        if (severity > reportable_severity_) {
            return;
        }
        auto logger = spdlog::get("app");
        if (!logger) {
            return;  // Logger not initialized yet
        }

        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                logger->error("[TensorRT] {}", msg);
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                logger->error("[TensorRT] {}", msg);
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                logger->warn("[TensorRT] {}", msg);
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                logger->info("[TensorRT] {}", msg);
                break;
            default:
                logger->debug("[TensorRT] {}", msg);
                break;
        }
    }
};

/**
 * @class LoggerManager
 * @brief 日志管理器 - 单例模式
 * 
 * 负责初始化和管理spdlog日志系统
 * 支持多个sink：控制台输出和文件保存（按日期和latest）
 * 支持CUDA日志通过spdlog集成
 */
class LoggerManager {
  public:
    /**
     * @brief 获取单例实例，初始化日志系统
     * @param config 配置管理器引用
     * @return LoggerManager单例引用
     */
    static LoggerManager & getInstance(ConfigManager & config) {
        static LoggerManager instance(config);
        return instance;
    }

    LoggerManager(const LoggerManager &)             = delete;
    LoggerManager & operator=(const LoggerManager &) = delete;

    /**
     * @brief 获取全局logger
     * @return spdlog::logger指针
     */
    std::shared_ptr<spdlog::logger> getLogger() const { return logger_; }

  private:
    explicit LoggerManager(ConfigManager & config);

    std::shared_ptr<spdlog::logger> logger_;

    /**
     * @brief 创建日期格式的日志文件名
     * @return 日期格式: "logs/YYYY-MM-DD.log"
     */
    static std::string getDateLogFilePath();

    /**
     * @brief 创建logs目录
     */
    static void createLogsDirectory();

    /**
     * @brief 将spdlog日志级别字符串转换为spdlog::level::level_enum
     * @param level_str 级别字符串: "trace"/"debug"/"info"/"warn"/"err"/"critical"
     * @return spdlog::level::level_enum
     */
    static spdlog::level::level_enum stringToLogLevel(const std::string & level_str);
};

// 便捷宏定义
#define APP_TRACE(...)    SPDLOG_LOGGER_TRACE(spdlog::get("app"), __VA_ARGS__)
#define APP_DEBUG(...)    SPDLOG_LOGGER_DEBUG(spdlog::get("app"), __VA_ARGS__)
#define APP_INFO(...)     SPDLOG_LOGGER_INFO(spdlog::get("app"), __VA_ARGS__)
#define APP_WARN(...)     SPDLOG_LOGGER_WARN(spdlog::get("app"), __VA_ARGS__)
#define APP_ERROR(...)    SPDLOG_LOGGER_ERROR(spdlog::get("app"), __VA_ARGS__)
#define APP_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::get("app"), __VA_ARGS__)
