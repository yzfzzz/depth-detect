#pragma once

#include "config_manager.h"

#include <NvInfer.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>

// TensorRT 日志适配器 - 将 TensorRT 日志写入 spdlog
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

// 日志管理器 - 单例模式
// 负责初始化和管理 spdlog 日志系统
// 支持多个 sink：控制台输出和文件保存（按日期和 latest）
class LoggerManager {
  public:
    // 获取单例实例，初始化日志系统
    // save_file: 是否保存日志文件
    // console_output: 是否在终端显示日志
    // log_level_str: 日志级别字符串 ("trace"/"debug"/"info"/"warn"/"err"/"critical")
    static LoggerManager & getInstance(bool                save_file,
                                       bool                console_output,
                                       const std::string & log_level_str) {
        static LoggerManager instance(save_file, console_output, log_level_str);
        return instance;
    }

    static LoggerManager & getInstance(ConfigManager & config_manager) {
        return getInstance(config_manager.isLogFileSaveEnabled(),
                           config_manager.isLogConsoleOutputEnabled(),
                           config_manager.getLogLevel());
    }

    LoggerManager(const LoggerManager &)             = delete;
    LoggerManager & operator=(const LoggerManager &) = delete;

    // 获取全局 logger
    std::shared_ptr<spdlog::logger> getLogger() const { return logger_; }

  private:
    explicit LoggerManager(bool save_file, bool console_output, const std::string & log_level_str);

    std::shared_ptr<spdlog::logger> logger_;

    // 创建日期格式的日志文件名，如 "logs/YYYY-MM-DD_HH-MM.log"
    static std::string getDateLogFilePath();

    // 创建 logs 目录
    static void createLogsDirectory();

    // 将日志级别字符串转换为 spdlog::level::level_enum
    static spdlog::level::level_enum stringToLogLevel(const std::string & level_str);
};

// 便捷宏定义
#define APP_TRACE(...)    SPDLOG_LOGGER_TRACE(spdlog::get("app"), __VA_ARGS__)
#define APP_DEBUG(...)    SPDLOG_LOGGER_DEBUG(spdlog::get("app"), __VA_ARGS__)
#define APP_INFO(...)     SPDLOG_LOGGER_INFO(spdlog::get("app"), __VA_ARGS__)
#define APP_WARN(...)     SPDLOG_LOGGER_WARN(spdlog::get("app"), __VA_ARGS__)
#define APP_ERROR(...)    SPDLOG_LOGGER_ERROR(spdlog::get("app"), __VA_ARGS__)
#define APP_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::get("app"), __VA_ARGS__)
