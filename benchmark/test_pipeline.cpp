#include "config_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

class PipelineBenchmark : public benchmark::Fixture {
  public:
    std::unique_ptr<Pipeline> pipeline;
    cv::VideoCapture          cap;
    cv::Mat                   img;
    int                       num_frames = 0;
    FrameMeta                 frame_meta;

    void SetUp(const ::benchmark::State & state) override {
        ConfigManager config_manager("config.yaml");
        cap.open("../data/shu/1shu_east_0514.mp4");  // 替换为你的测试视频路径
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video");
        }
        frame_meta =
            FrameMeta(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT),
                      cap.get(cv::CAP_PROP_FPS), FrameSource::VIDEO);
        pipeline = std::make_unique<Pipeline>(config_manager, frame_meta);

        // Warmup: 跑 5 帧让 GPU 预热
        for (int i = 0; i < 20; ++i) {
            FrameInputContext  warmup_ctx(i, frame_meta);
            InferOutputContext warmup_out;
            if (cap.read(warmup_ctx.raw_img) && !warmup_ctx.raw_img.empty()) {
                pipeline->process(warmup_ctx, warmup_out);
            }
        }
        // Warmup 后重置视频到开头
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        num_frames = 0;
    }

    virtual void TearDown(benchmark::State & state) override {
        pipeline.reset();  // 显式释放 Pipeline，触发 YoloDetectModel 等的析构函数
        cap.release();
    }
};

BENCHMARK_DEFINE_F(PipelineBenchmark, ProcessInference)(benchmark::State & state) {
    for (auto _ : state) {
        FrameInputContext frame_input_context(num_frames, frame_meta);
        if (!cap.read(frame_input_context.raw_img) || frame_input_context.raw_img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // 视频播完自动重头
            continue;
        }
        num_frames++;

        InferOutputContext infer_output_context;
        pipeline->process(frame_input_context, infer_output_context);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_DEFINE_F(PipelineBenchmark, ProcessOverlapInference)(benchmark::State & state) {
    for (auto _ : state) {
        FrameInputContext frame_input_context(num_frames, frame_meta);
        if (!cap.read(frame_input_context.raw_img) || frame_input_context.raw_img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // 视频播完自动重头
            continue;
        }
        num_frames++;

        InferOutputContext infer_output_context;
        pipeline->processOverlap(frame_input_context, infer_output_context);
    }
    state.SetItemsProcessed(state.iterations());
}

// ==================== YOLO 纯推理 Benchmark ====================
BENCHMARK_DEFINE_F(PipelineBenchmark, YoloOnlyInferenceAsync)(benchmark::State & state) {
    for (auto _ : state) {
        FrameInputContext frame_input_context(num_frames, frame_meta);
        if (!cap.read(frame_input_context.raw_img) || frame_input_context.raw_img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        num_frames++;

        // 仅调用 YOLO 推理（包含预处理 + TRT + CUDA NMS kernel），无后处理
        pipeline->getDetector().inferenceAsync(frame_input_context.raw_img);
        pipeline->getDetector().waitAsync();  // 同步等待推理完成
    }
    state.SetItemsProcessed(state.iterations());
}

// ==================== Depth 纯推理 Benchmark ====================
BENCHMARK_DEFINE_F(PipelineBenchmark, DepthOnlyInferenceAsync)(benchmark::State & state) {
    for (auto _ : state) {
        FrameInputContext frame_input_context(num_frames, frame_meta);
        if (!cap.read(frame_input_context.raw_img) || frame_input_context.raw_img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        num_frames++;

        // 仅调用 Depth 推理（包含预处理 + TRT），无后处理
        pipeline->getDepthModel().predictAsync(frame_input_context.raw_img);
        pipeline->getDepthModel().waitAsync();  // 同步等待推理完成
    }
    state.SetItemsProcessed(state.iterations());
}

// ==================== PostProcess 纯后处理 Benchmark ====================
BENCHMARK_DEFINE_F(PipelineBenchmark, PostProcess)(benchmark::State & state) {
    // 1. 准备固定的输入数据（避免 IO 和推理耗时干扰）
    FrameInputContext      fixed_input(0, frame_meta);
    InferOutputContext     fixed_output;
    std::vector<Detection> fixed_detections;

    if (!cap.read(fixed_input.raw_img) || fixed_input.raw_img.empty()) {
        state.SkipWithError("Failed to read frame for postProcess benchmark");
        return;
    }

    // 2. 执行一次完整的异步推理，获取真实的 Detections 结果
    pipeline->getDetector().inferenceAsync(fixed_input.raw_img);
    pipeline->getDepthModel().predictAsync(fixed_input.raw_img);
    pipeline->getDepthModel().waitAsync();
    pipeline->getDetector().waitAsync();
    fixed_detections = pipeline->getDetector().getInferResultAsync(fixed_input.raw_img);

    auto depth_result         = pipeline->getDepthModel().getPredictResultAsync();
    fixed_output.result_depth = depth_result.first;
    fixed_output.depth_vis    = depth_result.second;

    // 3. 纯后处理循环测试
    for (auto _ : state) {
        InferOutputContext temp_output;
        temp_output.result_depth = fixed_output.result_depth;  // 传递深度图供运动状态计算
        temp_output.depth_vis = fixed_output.depth_vis;

        // 仅执行 postProcess（包含 ByteTrack 跟踪 + 运动状态引擎）
        pipeline->postProcess(fixed_input, temp_output, fixed_detections);

        benchmark::DoNotOptimize(temp_output);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(PipelineBenchmark, YoloOnlyInferenceAsync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);
BENCHMARK_REGISTER_F(PipelineBenchmark, DepthOnlyInferenceAsync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);
BENCHMARK_REGISTER_F(PipelineBenchmark, ProcessInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);
BENCHMARK_REGISTER_F(PipelineBenchmark, ProcessOverlapInference)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);
BENCHMARK_REGISTER_F(PipelineBenchmark, PostProcess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);
BENCHMARK_MAIN();
