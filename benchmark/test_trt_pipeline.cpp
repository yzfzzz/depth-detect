#include "config_manager.h"
#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

const char * video_path  = "../data/1shu_east_0514.mp4";
const char * config_path = "config.yaml";

// 全局单例：配置、日志、IO、流水线
ConfigManager     config_manager(config_path);
static const auto _ = []() -> bool {
    config_manager.setUseGPU(true);
    config_manager.setLogLevel("err");
    return true;
}();

LoggerManager & logger_manager = LoggerManager::getInstance(config_manager);
IOManager       io_manager(config_manager);
FrameMeta       frame_meta = io_manager.Init(video_path);
Pipeline        pipeline(config_manager, frame_meta);

// 对比同步串行 (process) 与 CPU/GPU 重叠 (processOverlap) 的性能
class PipelineBenchmark : public benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State & state) override {
        io_manager.Init(video_path);
        FrameInputContext  warmup_ctx(0, frame_meta);
        InferOutputContext warmup_out;
        // 预热 20 帧，消除首次推理的冷启动偏差
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
    // 通用 Benchmark 循环：读帧 → 计时 → 执行 → 循环播放
    template <typename Func> void RunPipelineBench(benchmark::State & state, Func && process_fn) {
        for (auto _ : state) {
            state.PauseTiming();
            FrameInputContext ctx(num_frames_, frame_meta);
            if (!io_manager.readNextFrame(ctx, false) || ctx.raw_img.empty()) {
                io_manager.Init(video_path);  // 播完循环
                continue;
            }
            num_frames_++;
            InferOutputContext out;
            state.ResumeTiming();

            process_fn(ctx, out, state);
        }
        state.SetItemsProcessed(state.iterations());
    }

    int num_frames_ = 0;
};

// ---- 端到端流水线 ----

// 同步串行：YOLO → Depth → ByteTrack → PostProcess
BENCHMARK_DEFINE_F(PipelineBenchmark, Process)(benchmark::State & state) {
    RunPipelineBench(state,
                     [](auto & ctx, auto & out, auto & state) { pipeline.process(ctx, out); });
}

// CPU/GPU 重叠：YOLO 与 Depth 异步并行
BENCHMARK_DEFINE_F(PipelineBenchmark, ProcessOverlap)(benchmark::State & state) {
    RunPipelineBench(
        state, [](auto & ctx, auto & out, auto & state) { pipeline.processOverlap(ctx, out); });
}

// ---- YOLO 检测各阶段 ----

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloInferenceAsync)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();
        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
        state.ResumeTiming();
        pipeline.detector_.runInferenceAsync(ctx);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, benchmark::State & state) {
        state.PauseTiming();
        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
        state.ResumeTiming();
        pipeline.detector_.runInference(ctx, out);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();

        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.runInferenceAsync(ctx);
        pipeline.detector_.synchronizeStream();

        state.ResumeTiming();

        pipeline.detector_.cudaPostProcess(ctx);
        pipeline.detector_.getInferOutputResult(out);
    });
}

// ---- 深度估计各阶段 ----

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthInferenceAsync)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();
        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.depth_model_.synchronizeStream();
        state.ResumeTiming();

        pipeline.depth_model_.runInferenceAsync(ctx);
        pipeline.depth_model_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto state) {
        state.PauseTiming();
        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.depth_model_.synchronizeStream();
        state.ResumeTiming();

        pipeline.depth_model_.runInference(ctx, out);
        pipeline.depth_model_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();

        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.depth_model_.runInferenceAsync(ctx);
        pipeline.depth_model_.synchronizeStream();

        state.ResumeTiming();

        pipeline.depth_model_.cudaPostProcess(ctx);
        pipeline.depth_model_.getInferOutputResult(out);
    });
}

// ---- 后处理 ----

// 测量 MotionState + PostProcess 耗时（不含推理），用 processOverlap 准备好推理结果
BENCHMARK_DEFINE_F(PipelineBenchmark, MotionStateEnginePostprocess)(benchmark::State & state) {
    for (auto _ : state) {
        state.PauseTiming();
        FrameInputContext ctx(num_frames_, frame_meta);
        if (!io_manager.readNextFrame(ctx, false) || ctx.raw_img.empty()) {
            io_manager.Init(video_path);
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
// 注册

BENCHMARK_REGISTER_F(PipelineBenchmark, ProcessOverlap)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/TensorRT/ProcessOverlap(Async)");

BENCHMARK_REGISTER_F(PipelineBenchmark, Process)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/TensorRT/Process(Sync)");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/CUDA/YoloPreprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/TensorRT/YoloInference");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloInferenceAsync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/TensorRT/YoloInferenceAsync");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/CUDA/YoloPostprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/CUDA/DepthPreprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthInferenceAsync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/TensorRT/DepthInferenceAsync");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/TensorRT/DepthInference");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/CUDA/DepthPostprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, MotionStateEnginePostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/MotionStateEnginePostprocess");

BENCHMARK_MAIN();
