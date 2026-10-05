#pragma once

#include "config_manager.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// 单个 track 单帧的记录（运行结束时统一写入 CSV）
struct TrackRecord {
    int   frame_id;
    int   class_id;
    float x;  // bbox 左上角 x
    float w;  // bbox 宽
    float y;  // bbox 左上角 y
    float h;  // bbox 高
    float area;
    float raw_depth;
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

    // 打印 ConfigManager 中的所有配置项
    static void logConfig(const ConfigManager & config);

    // 把累积的 track 记录统一写入 CSV（运行结束时调用；失败时回退到当前目录）
    static void saveTrackCsv(const std::string &                                       path,
                             const std::unordered_map<int, std::vector<TrackRecord>> & track_log);

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
