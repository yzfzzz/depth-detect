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
ConfigManager & config_manager = []() -> ConfigManager & {
    auto & cm = ConfigManager::getInstance(config_path);
    cm.setUseGPU(false);
    cm.setLogLevel("err");
    return cm;
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

// ---- YOLO 检测各阶段 ----

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        pipeline.detector_.cvMatPreProcess(ctx);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, benchmark::State & state) {
        state.PauseTiming();
        pipeline.detector_.cvMatPreProcess(ctx);
        state.ResumeTiming();
        pipeline.detector_.runInference(ctx, out);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, YoloPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();

        pipeline.detector_.cvMatPreProcess(ctx);
        pipeline.detector_.runInference(ctx, out);

        state.ResumeTiming();

        pipeline.detector_.cvMatPostProcess(out);
    });
}

// ---- 深度估计各阶段 ----

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        pipeline.depth_model_.cvMatPreProcess(ctx);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto state) {
        state.PauseTiming();
        pipeline.depth_model_.cvMatPreProcess(ctx);
        state.ResumeTiming();

        pipeline.depth_model_.runInference(ctx, out);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, DepthPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();

        pipeline.depth_model_.cvMatPreProcess(ctx);
        pipeline.depth_model_.runInference(ctx, out);

        state.ResumeTiming();

        pipeline.depth_model_.cvMatPostProcess(out);
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
        pipeline.process(ctx, out);
        state.ResumeTiming();
        pipeline.postProcess(ctx, out);
    }
    state.SetItemsProcessed(state.iterations());
}

// ============================================================================
// 注册

BENCHMARK_REGISTER_F(PipelineBenchmark, Process)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/Onnx/Process(Sync)");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/OpenCV2/YoloPreprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/Onnx/YoloInference");

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/OpenCV2/YoloPostprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/OpenCV2/DepthPreprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/Onnx/DepthInference");

BENCHMARK_REGISTER_F(PipelineBenchmark, DepthPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/OpenCV2/DepthPostprocess");

BENCHMARK_REGISTER_F(PipelineBenchmark, MotionStateEnginePostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100)
    ->Name("Pipeline/MotionStateEnginePostprocess");

BENCHMARK_MAIN();
