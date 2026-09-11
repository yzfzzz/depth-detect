#include "logger_manager.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>

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
        std::string path   = "logs";
        std::string cmd    = "mkdir -p " + path;
        int         result = system(cmd.c_str());
        if (result != 0) {
            fprintf(stderr, "Failed to create logs directory, system() returned: %d\n", result);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "Failed to create logs directory: %s\n", e.what());
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

LoggerManager::LoggerManager(bool                save_file,
                             bool                console_output,
                             const std::string & log_level_str) {
    // 创建sinks容器
    std::vector<spdlog::sink_ptr> sinks;
    auto                          log_level = stringToLogLevel(log_level_str);

    // 文件输出sinks（按日期和latest）
    if (save_file) {
        createLogsDirectory();

        // 按日期的日志文件
        try {
            std::string date_log_path = getDateLogFilePath();
            auto        file_sink =
                std::make_shared<spdlog::sinks::basic_file_sink_mt>(date_log_path, true);
            file_sink->set_level(log_level);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%s:%#] %v");
            sinks.push_back(file_sink);
        } catch (const spdlog::spdlog_ex & ex) {
            fprintf(stderr, "File sink creation failed: %s\n", ex.what());
        }

        // latest.log文件（始终覆盖）
        try {
            auto latest_sink =
                std::make_shared<spdlog::sinks::basic_file_sink_mt>("latest.log", true);
            latest_sink->set_level(log_level);
            latest_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%s:%#] %v");
            sinks.push_back(latest_sink);
        } catch (const spdlog::spdlog_ex & ex) {
            fprintf(stderr, "Latest sink creation failed: %s\n", ex.what());
        }
    }

    // 如果没有任何sink或指定控制台输出，则添加控制台sink
    if (sinks.empty() || console_output) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%s:%#] %v");
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

void LoggerManager::saveTrackCsv(
    const std::string &                                       path,
    const std::unordered_map<int, std::vector<TrackRecord>> & track_log) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        out.open("track_log.csv", std::ios::out | std::ios::trunc);  // 回退到当前目录
    }
    if (!out.is_open()) {
        APP_WARN("Failed to open track log CSV");
        return;
    }

    out << "track_id,frame_id,class_id,x,w,y,h,area,raw_depth\n";

    // unordered_map 无序，按 track_id 排序后输出，保证结果可复现、便于阅读
    std::vector<int> ids;
    ids.reserve(track_log.size());
    for (const auto & kv : track_log) {
        ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    for (const int id : ids) {
        for (const auto & r : track_log.at(id)) {
            out << id << ',' << r.frame_id << ',' << r.class_id << ',' << r.x << ',' << r.w << ','
                << r.y << ',' << r.h << ',' << r.area << ',' << r.raw_depth << '\n';
        }
    }
    out.close();
    APP_INFO("Track log saved to {} ({} tracks)", path, ids.size());
}

void LoggerManager::logConfig(const ConfigManager & config) {
    APP_INFO("========== ConfigManager Settings ==========");

    // display_manager
    APP_INFO("  [display_manager] is_display: {}", config.isDisplayEnabled());

    // prefer
    APP_INFO("  [prefer] use_gpu: {}", config.isUseGPU());
    APP_INFO("  [prefer] overlap: {}", config.isOverlapEnabled());

    // yolo
    auto yolo_models = config.getYoloModelPath();
    for (const auto & kv : yolo_models) {
        APP_INFO("  [yolo] model_path[{}]: {}", kv.first, kv.second);
    }
    APP_INFO("  [yolo] nms_thresh: {}", config.getYoloNmsThresh());
    APP_INFO("  [yolo] conf_thresh: {}", config.getYoloConfThresh());

    // depth
    auto depth_models = config.getDepthModelPath();
    for (const auto & kv : depth_models) {
        APP_INFO("  [depth] model_path[{}]: {}", kv.first, kv.second);
    }
    APP_INFO("  [depth] depth_interval: {}", config.getDepthInterval());

    // motion_state_engine：快速靠近（approach）检测参数（原 TTC/运动状态一路已移除）
    APP_INFO("  [motion_state_engine.approach] enabled: {}", config.isApproachEnabled());
    APP_INFO("  [motion_state_engine.approach] filter: {}", config.getApproachFilterMode());
    APP_INFO("  [motion_state_engine.approach] warmup/recent_w: {}/{}", config.getApproachWarmup(),
             config.getApproachRecentW());
    APP_INFO("  [motion_state_engine.approach] thr_depth/thr_height: {}/{}",
             config.getApproachThrDepth(), config.getApproachThrHeight());
    APP_INFO("  [motion_state_engine.approach] score_thr/confirm: {}/{}",
             config.getApproachScoreThr(), config.getApproachConfirm());
    APP_INFO("  [motion_state_engine.approach] exit_score_thr/exit_confirm: {}/{}",
             config.getApproachExitScoreThr(), config.getApproachExitConfirm());

    // danger_alert
    APP_INFO("  [danger_alert] is_filter_small_objects: {}", config.isFilterSmallObjectsEnabled());
    APP_INFO("  [danger_alert] min_object_area: {}", config.getMinObjectArea());

    // io_manager
    APP_INFO("  [io_manager] out_dir: {}", config.getOutDir());
    APP_INFO("  [io_manager] save_mode: {}", config.getSaveMode());
    APP_INFO("  [io_manager] is_save: {}", config.isSaveEnabled());
    APP_INFO("  [io_manager] send_tcp: {}", config.isSendTcpEnabled());
    APP_INFO("  [io_manager] send_tcp_ip: {}", config.getSendTcpIp());
    APP_INFO("  [io_manager] send_tcp_port: {}", config.getSendTcpPort());

    // logger
    APP_INFO("  [logger] save_file: {}", config.isLogFileSaveEnabled());
    APP_INFO("  [logger] console_output: {}", config.isLogConsoleOutputEnabled());
    APP_INFO("  [logger] log_level: {}", config.getLogLevel());

    APP_INFO("==============================================");
}
