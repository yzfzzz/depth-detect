#include "config_manager.h"
#include "depth_model.h"
#include "preprocess.h"

#include <benchmark/benchmark.h>

#include <memory>
#include <opencv2/opencv.hpp>

class PreprocessBenchmark : public benchmark::Fixture {
  public:
    std::unique_ptr<DepthModel> depth_model;
    cv::VideoCapture            cap;
    cv::Mat                     img;
    int                         num_frames = 0;

    int raw_w = 0, raw_h = 0;
    int model_w = 640, model_h = 192;

    void SetUp(const ::benchmark::State & state) override {
        ConfigManager config_manager("config.yaml");

        cap.open("../data/shu/1shu_east_0514.mp4");
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video");
        }

        raw_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        raw_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // 初始化模型
        depth_model       = std::make_unique<DepthModel>();
        bool is_normalize = false;
        if (config_manager.getDepthEnginePath().find("depth_anything") != std::string::npos) {
            is_normalize = true;
        }
        depth_model->init(config_manager.getDepthEnginePath(), raw_w, raw_h, is_normalize);

        // Warmup: 读前 20 帧并预热
        for (int i = 0; i < 20; ++i) {
            if (cap.read(img) && !img.empty()) {
                // CPU warmup
                auto out = depth_model->preProcess(img);

                // GPU warmup
                depth_model->preProcessAsync(img);
                depth_model->waitAsync();
            }
        }

        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        num_frames = 0;
    }

    void TearDown(const ::benchmark::State & state) override {
        cap.release();
        depth_model.reset();  // DepthModel 析构时自动释放所有 GPU 资源
    }
};

// CPU Preprocess
BENCHMARK_DEFINE_F(PreprocessBenchmark, CPU_Preprocess)(benchmark::State & state) {
    for (auto _ : state) {
        state.PauseTiming();
        if (!cap.read(img) || img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        num_frames++;
        state.ResumeTiming();

        // CPU 前处理
        std::vector<float> input = depth_model->preProcess(img);
        benchmark::DoNotOptimize(input);
    }
    state.SetItemsProcessed(state.iterations());
}

// GPU Async Preprocess
BENCHMARK_DEFINE_F(PreprocessBenchmark, GPU_Async_Preprocess)(benchmark::State & state) {
    for (auto _ : state) {
        state.PauseTiming();
        if (!cap.read(img) || img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        num_frames++;
        state.ResumeTiming();

        // GPU 异步前处理
        depth_model->preProcessAsync(img);
        depth_model->waitAsync();
        benchmark::DoNotOptimize(depth_model);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(PreprocessBenchmark, CPU_Preprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PreprocessBenchmark, GPU_Async_Preprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_MAIN();
