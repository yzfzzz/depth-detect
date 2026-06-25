#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

std::string config_path      = "benchmark.yaml";
std::string task_name        = "test_trt_pipeline";
YAML::Node  root             = YAML::LoadFile(config_path);
std::string video_path       = root["video_path"].as<std::string>();
YAML::Node  task_node        = root["task"][task_name];
std::string yolo_model_path  = task_node["yolo_model_path"].as<std::string>();
std::string depth_model_path = task_node["depth_model_path"].as<std::string>();

LoggerManager & logger_manager = LoggerManager::getInstance(false, true, "err");
IOManager       io_manager("none");
FrameMeta       frame_meta = io_manager.Init(video_path);
Pipeline        pipeline(depth_model_path, yolo_model_path, frame_meta, true);

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
BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_TensorRT_Process_Sync)(benchmark::State & state) {
    RunPipelineBench(state,
                     [](auto & ctx, auto & out, auto & s) { pipeline.process(ctx, out); });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_TensorRT_ProcessOverlap_Async)(benchmark::State & state) {
    RunPipelineBench(
        state, [](auto & ctx, auto & out, auto & s) { pipeline.processOverlap(ctx, out); });
}

// ---- YOLO 检测各阶段 ----
BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_CUDA_YoloPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_TensorRT_YoloInferenceAsync)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        s.PauseTiming();
        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
        s.ResumeTiming();
        pipeline.detector_.runInferenceAsync(ctx);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_TensorRT_YoloInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        s.PauseTiming();
        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
        s.ResumeTiming();
        pipeline.detector_.runInference(ctx, out);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_CUDA_YoloPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        s.PauseTiming();
        pipeline.detector_.cudaPreProcess(ctx);
        pipeline.detector_.runInferenceAsync(ctx);
        pipeline.detector_.synchronizeStream();
        s.ResumeTiming();
        pipeline.detector_.cudaPostProcess(ctx);
        pipeline.detector_.getInferOutputResult(out);
    });
}

// ---- 深度估计各阶段 ----
BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_CUDA_DepthPreprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.detector_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_TensorRT_DepthInferenceAsync)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        s.PauseTiming();
        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.depth_model_.synchronizeStream();
        s.ResumeTiming();
        pipeline.depth_model_.runInferenceAsync(ctx);
        pipeline.depth_model_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_TensorRT_DepthInference)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        s.PauseTiming();
        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.depth_model_.synchronizeStream();
        s.ResumeTiming();
        pipeline.depth_model_.runInference(ctx, out);
        pipeline.depth_model_.synchronizeStream();
    });
}

BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_CUDA_DepthPostprocess)(benchmark::State & state) {
    RunPipelineBench(state, [](auto & ctx, auto & out, auto & s) {
        s.PauseTiming();
        pipeline.depth_model_.cudaPreProcess(ctx);
        pipeline.depth_model_.runInferenceAsync(ctx);
        pipeline.depth_model_.synchronizeStream();
        s.ResumeTiming();
        pipeline.depth_model_.cudaPostProcess(ctx);
        pipeline.depth_model_.getInferOutputResult(out);
    });
}

// ---- 后处理 ----
BENCHMARK_DEFINE_F(PipelineBenchmark, Pipeline_MotionStateEnginePostprocess)(benchmark::State & state) {
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
// 注册（无 ->Name，名称由 BENCHMARK_DEFINE_F 的宏名决定）
// ============================================================================
BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_TensorRT_ProcessOverlap_Async)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_TensorRT_Process_Sync)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_CUDA_YoloPreprocess)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_TensorRT_YoloInference)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_TensorRT_YoloInferenceAsync)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_CUDA_YoloPostprocess)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_CUDA_DepthPreprocess)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_TensorRT_DepthInferenceAsync)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_TensorRT_DepthInference)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_CUDA_DepthPostprocess)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_REGISTER_F(PipelineBenchmark, Pipeline_MotionStateEnginePostprocess)
    ->Unit(benchmark::kMillisecond)->Iterations(100);

BENCHMARK_MAIN();