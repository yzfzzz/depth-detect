#include "config_manager.h"
#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

const char * video_path  = "../data/1shu_east_0514.mp4";
const char * config_path = "config.yaml";

// 读取配置文件 - 使用单例模式
ConfigManager & config_manager = ConfigManager::getInstance(config_path);
// 初始化日志系统
LoggerManager & logger_manager = LoggerManager::getInstance(config_manager);
// 文件读写，落盘保存, 以及视频读取（包括模拟相机延迟）
IOManager       io_manager(config_manager);
FrameMeta       frame_meta = io_manager.Init(video_path);
// 推理流水线（负责目标检测、深度估计、跟踪、运动状态判断等核心功能）
Pipeline        pipeline(config_manager, frame_meta);

// PipelineBenchmark — 对比 process()（同步） vs processOverlap()（CPU/GPU重叠）
class PipelineBenchmark : public benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State & state) override {
        io_manager.Init(video_path);
        FrameInputContext  warmup_ctx(0, frame_meta);
        InferOutputContext warmup_out;
        for (int i = 0; i < 20; ++i) {
            if (!io_manager.readNextFrame(warmup_ctx, false) || warmup_ctx.raw_img.empty()) {
                break;
            }
            pipeline.process(warmup_ctx, warmup_out);
            pipeline.processOverlap(warmup_ctx, warmup_out);
        }
        num_frames_ = 0;
    }

  protected:
    // 公共 Benchmark 循环：封装帧读取、循环播放、计时控制
    template <typename Func> void RunPipelineBench(benchmark::State & state, Func && process_fn) {
        for (auto _ : state) {
            state.PauseTiming();
            FrameInputContext ctx(num_frames_, frame_meta);
            if (!io_manager.readNextFrame(ctx, false) || ctx.raw_img.empty()) {
                io_manager.Init(video_path);  // 播完从头循环
                continue;
            }
            num_frames_++;
            InferOutputContext out;
            state.ResumeTiming();

            process_fn(ctx, out);
        }
        state.SetItemsProcessed(state.iterations());
    }

    int num_frames_ = 0;
};

// 1. process — 同步串行：YOLO → Depth → ByteTrack → PostProcess
BENCHMARK_DEFINE_F(PipelineBenchmark, Process)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) { pipeline.process(ctx, out); });
}

// 2. processOverlap — CPU/GPU 重叠：YOLO 与 Depth 异步并行
BENCHMARK_DEFINE_F(PipelineBenchmark, ProcessOverlap)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) { pipeline.processOverlap(ctx, out); });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) {
        // 仅测 YOLO 预处理 (preprocess + H2D)
        pipeline.detector_.cudaPreProcess(ctx);  // 需要把 preprocess 暴露为 public
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) {
        pipeline.detector_.runInference(ctx, out);  // 纯推理
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) {
        pipeline.detector_.cudaPostProcess(ctx);  // 纯推理
        pipeline.detector_.getInferOutputResult(out);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) {
        pipeline.depth_model_.cudaPostProcess(ctx);  // 纯推理
        pipeline.depth_model_.getInferOutputResult(out);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) {
        pipeline.depth_model_.runInference(ctx, out);  // 纯推理
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out) {
        pipeline.depth_model_.cudaPostProcess(ctx);  // 纯推理
        pipeline.depth_model_.getInferOutputResult(out);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, MotionStateEnginePostprocess)(benchmark::State & state) {
    for (auto _ : state) {
        state.PauseTiming();
        FrameInputContext ctx(num_frames_, frame_meta);
        if (!io_manager.readNextFrame(ctx, false) || ctx.raw_img.empty()) {
            io_manager.Init(video_path);  // 播完从头循环
            continue;
        }
        num_frames_++;
        InferOutputContext out;
        pipeline.processOverlap(ctx, out);
        state.ResumeTiming();
        pipeline.postProcess(ctx, out);
    }
    state.SetItemsProcessed(state.iterations());
}

// ============================================================================
BENCHMARK_REGISTER_F(PipelineBenchmark, ProcessOverlap)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/ProcessOverlap(Async)");
BENCHMARK_REGISTER_F(PipelineBenchmark, Process)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/Process(Sync)");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/cuda/YoloPreprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/cuda/YoloInference");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/cuda/YoloPostprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/cuda/DepthPreprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/cuda/DepthInference");
BENCHMARK_REGISTER_F(PipelineBenchmark, DepthPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/cuda/DepthPostprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, MotionStateEnginePostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/MotionStateEnginePostprocess");

BENCHMARK_MAIN();
