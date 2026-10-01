#pragma once

#include <NvInfer.h>
#include <spdlog/spdlog.h>

// TensorRT 日志适配器 - 将 TensorRT 日志写入 spdlog
// 仅 TensorRT 后端使用，避免 TRT 头文件经 logger_manager.h 扩散到全工程
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
