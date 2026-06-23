#include "config_manager.h"
#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

std::string config_path      = "benchmark.yaml";
std::string task_name        = "test_onnx_pipeline";
YAML::Node  root             = YAML::LoadFile(config_path);           // 先拿到根节点
std::string video_path       = root["video_path"].as<std::string>();  // 根层级读 video_path
YAML::Node  task_node        = root["task"][task_name];
std::string yolo_model_path  = task_node["yolo_model_path"].as<std::string>();
std::string depth_model_path = task_node["depth_model_path"].as<std::string>();

LoggerManager & logger_manager = LoggerManager::getInstance(false, true, "err");
IOManager       io_manager("none");
FrameMeta       frame_meta = io_manager.Init(video_path);
Pipeline        pipeline(depth_model_path, yolo_model_path, frame_meta, false);

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
