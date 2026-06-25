#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

std::string config_path      = "benchmark.yaml";
std::string task_name        = "test_onnx_pipeline";
YAML::Node  root             = YAML::LoadFile(config_path);
std::string video_path       = root["video_path"].as<std::string>();
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
                io_manager.Init(video_path);
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

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_Onnx_Process_Sync)(benchmark::State & state) {
    RunPipelineBench(state,
                     [](auto & ctx, auto & out, auto & state) { pipeline.process(ctx, out); });
}

// ---- YOLO 检测各阶段 ----

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_OpenCV2_YoloPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        pipeline.detector_.cvMatPreProcess(ctx);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_Onnx_YoloInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();
        pipeline.detector_.cvMatPreProcess(ctx);
        state.ResumeTiming();
        pipeline.detector_.runInference(ctx, out);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_OpenCV2_YoloPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();

        pipeline.detector_.cvMatPreProcess(ctx);
        pipeline.detector_.runInference(ctx, out);

        state.ResumeTiming();

        pipeline.detector_.cvMatPostProcess(out);
    });
}

// ---- 深度估计各阶段 ----

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_OpenCV2_DepthPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        pipeline.depth_model_.cvMatPreProcess(ctx);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_Onnx_DepthInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();
        pipeline.depth_model_.cvMatPreProcess(ctx);
        state.ResumeTiming();

        pipeline.depth_model_.runInference(ctx, out);
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_OpenCV2_DepthPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & state) {
        state.PauseTiming();

        pipeline.depth_model_.cvMatPreProcess(ctx);
        pipeline.depth_model_.runInference(ctx, out);

        state.ResumeTiming();

        pipeline.depth_model_.cvMatPostProcess(out);
    });
}

// ---- 后处理 ----

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_MotionStateEnginePostprocess)
(benchmark::State & state) {
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
// 注册（无 ->Name，1.7 兼容）
// ============================================================================

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_Onnx_Process_Sync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_OpenCV2_YoloPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_Onnx_YoloInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_OpenCV2_YoloPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_OpenCV2_DepthPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_Onnx_DepthInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_OpenCV2_DepthPostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_MotionStateEnginePostprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_MAIN();
