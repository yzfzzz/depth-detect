#include "logger_manager.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

std::string LoggerManager::getDateLogFilePath() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto tm   = std::localtime(&time);

    std::ostringstream oss;
    oss << "logs/" << std::put_time(tm, "%Y-%m-%d_%H-%M") << ".log";
    return oss.str();
}

void LoggerManager::createLogsDirectory() {
    try {
        std::string path = "logs";
        std::string cmd  = "mkdir -p " + path;
        system(cmd.c_str());
    } catch (const std::exception & e) {
        std::cerr << "Failed to create logs directory: " << e.what() << std::endl;
    }
}

spdlog::level::level_enum LoggerManager::stringToLogLevel(const std::string & level_str) {
    static const std::unordered_map<std::string, spdlog::level::level_enum> log_level_map = {
        { "trace",    spdlog::level::trace    },
        { "debug",    spdlog::level::debug    },
        { "info",     spdlog::level::info     },
        { "warn",     spdlog::level::warn     },
        { "err",      spdlog::level::err      },
        { "critical", spdlog::level::critical }
    };

    auto it = log_level_map.find(level_str);
    if (it != log_level_map.end()) {
        return it->second;
    }
    return spdlog::level::info;
}

LoggerManager::LoggerManager(const ConfigManager & config) {
    // 创建sinks容器
    std::vector<spdlog::sink_ptr> sinks;

    // 获取日志配置
    bool        save_file      = config.isLogFileSaveEnabled();
    bool        console_output = config.isLogConsoleOutputEnabled();
    std::string log_level_str  = config.getLogLevel();
    auto        log_level      = stringToLogLevel(log_level_str);

    // 文件输出sinks（按日期和latest）
    if (save_file) {
        createLogsDirectory();

        // 按日期的日志文件
        try {
            std::string date_log_path = getDateLogFilePath();
            auto        file_sink =
                std::make_shared<spdlog::sinks::basic_file_sink_mt>(date_log_path, true);
            file_sink->set_level(log_level);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
            sinks.push_back(file_sink);
        } catch (const spdlog::spdlog_ex & ex) {
            std::cerr << "File sink creation failed: " << ex.what() << std::endl;
        }

        // latest.log文件（始终覆盖）
        try {
            auto latest_sink =
                std::make_shared<spdlog::sinks::basic_file_sink_mt>("latest.log", true);
            latest_sink->set_level(log_level);
            latest_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
            sinks.push_back(latest_sink);
        } catch (const spdlog::spdlog_ex & ex) {
            std::cerr << "Latest sink creation failed: " << ex.what() << std::endl;
        }
    }

    // 如果没有任何sink或指定控制台输出，则添加控制台sink
    if (sinks.empty() || console_output) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
        sinks.push_back(console_sink);
    }

    // 创建logger
    logger_ = std::make_shared<spdlog::logger>("app", sinks.begin(), sinks.end());
    logger_->set_level(log_level);
    logger_->flush_on(spdlog::level::err);

    // 注册为全局logger
    spdlog::register_logger(logger_);
    spdlog::set_default_logger(logger_);

    APP_INFO("Logger initialized successfully");
    APP_INFO("Log level: {}", log_level_str);
    if (save_file) {
        APP_INFO("Log files saved to: logs/{} and latest.log",
                 std::string(getDateLogFilePath()).substr(5));
    }
    if (console_output) {
        APP_INFO("Console output enabled");
    }
}
