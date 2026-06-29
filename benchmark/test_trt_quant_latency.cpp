#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

// 全局配置：三套 Pipeline（INT8 / FP16 / FP32）
std::string config_path = "benchmark.yaml";
std::string task_name   = "test_quant_latency";
YAML::Node  root        = YAML::LoadFile(config_path);
std::string video_path  = root["video_path"].as<std::string>();
YAML::Node  task_node   = root["task"][task_name];

LoggerManager & logger_manager = LoggerManager::getInstance(false, true, "err");
IOManager       io_manager("none");
FrameMeta       frame_meta = io_manager.Init(video_path);
#ifndef __aarch64__
std::string yolo_int8_path  = task_node["yolo_int8_path"].as<std::string>();
std::string depth_int8_path = task_node["depth_int8_path"].as<std::string>();
Pipeline    g_int8_pipe(depth_int8_path, yolo_int8_path, frame_meta, true);
#endif
std::string yolo_fp16_path  = task_node["yolo_fp16_path"].as<std::string>();
std::string depth_fp16_path = task_node["depth_fp16_path"].as<std::string>();
std::string yolo_fp32_path  = task_node["yolo_fp32_path"].as<std::string>();
std::string depth_fp32_path = task_node["depth_fp32_path"].as<std::string>();
Pipeline    g_fp16_pipe(depth_fp16_path, yolo_fp16_path, frame_meta, true);
Pipeline    g_fp32_pipe(depth_fp32_path, yolo_fp32_path, frame_meta, true);

// Fixture：量化延迟 Benchmark 共享基类，负责预热和帧迭代
class QuantLatencyBench : public benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State & state) override {
        io_manager.Init(video_path);
        FrameInputContext  warmup_ctx(0, frame_meta);
        InferOutputContext warmup_out;
        // 三套精度各预热 20 帧
        for (int i = 0; i < 20; ++i) {
            if (!io_manager.readNextFrame(warmup_ctx, false) || warmup_ctx.raw_img.empty()) {
                break;
            }
#ifndef __aarch64__
            g_int8_pipe.process(warmup_ctx, warmup_out);
            g_int8_pipe.processOverlap(warmup_ctx, warmup_out);
#endif
            g_fp16_pipe.process(warmup_ctx, warmup_out);
            g_fp16_pipe.processOverlap(warmup_ctx, warmup_out);
            g_fp32_pipe.process(warmup_ctx, warmup_out);
            g_fp32_pipe.processOverlap(warmup_ctx, warmup_out);
        }
        num_frames_ = 0;
    }

  protected:
    template <typename Func> void run(benchmark::State & state, Func && fn) {
        for (auto _ : state) {
            state.PauseTiming();
            FrameInputContext ctx(num_frames_, frame_meta);
            if (!io_manager.readNextFrame(ctx, false) || ctx.raw_img.empty()) {
                io_manager.Init(video_path);
                continue;
            }
            num_frames_++;
            InferOutputContext out;
            state.ResumeTiming();
            fn(ctx, out);
        }
        state.SetItemsProcessed(state.iterations());
    }

    int num_frames_ = 0;
};

// Benchmarks：各精度/模式延迟对比

// INT8 精度
#ifndef __aarch64__
BENCHMARK_DEFINE_F(QuantLatencyBench, QuantLatency_INT8_Process_Sync)(benchmark::State & state) {
    run(state, [](auto & ctx, auto & out) { g_int8_pipe.process(ctx, out); });
}

BENCHMARK_DEFINE_F(QuantLatencyBench, QuantLatency_INT8_Process_Overlap)(benchmark::State & state) {
    run(state, [](auto & ctx, auto & out) { g_int8_pipe.processOverlap(ctx, out); });
}
#endif

// FP16 精度
BENCHMARK_DEFINE_F(QuantLatencyBench, QuantLatency_FP16_Process_Sync)(benchmark::State & state) {
    run(state, [](auto & ctx, auto & out) { g_fp16_pipe.process(ctx, out); });
}

BENCHMARK_DEFINE_F(QuantLatencyBench, QuantLatency_FP16_Process_Overlap)(benchmark::State & state) {
    run(state, [](auto & ctx, auto & out) { g_fp16_pipe.processOverlap(ctx, out); });
}

// FP32 精度
BENCHMARK_DEFINE_F(QuantLatencyBench, QuantLatency_FP32_Process_Sync)(benchmark::State & state) {
    run(state, [](auto & ctx, auto & out) { g_fp32_pipe.process(ctx, out); });
}

BENCHMARK_DEFINE_F(QuantLatencyBench, QuantLatency_FP32_Process_Overlap)(benchmark::State & state) {
    run(state, [](auto & ctx, auto & out) { g_fp32_pipe.processOverlap(ctx, out); });
}

// 注册 Benchmark（名称由 BENCHMARK_DEFINE_F 宏名决定）
#ifndef __aarch64__
BENCHMARK_REGISTER_F(QuantLatencyBench, QuantLatency_INT8_Process_Sync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(QuantLatencyBench, QuantLatency_INT8_Process_Overlap)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);
#endif

BENCHMARK_REGISTER_F(QuantLatencyBench, QuantLatency_FP16_Process_Sync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(QuantLatencyBench, QuantLatency_FP16_Process_Overlap)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(QuantLatencyBench, QuantLatency_FP32_Process_Sync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(QuantLatencyBench, QuantLatency_FP32_Process_Overlap)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_MAIN();
