// 三种调度策略（串行 SYNC / 重叠 OVERLAP / 错峰 STAGGER）的单帧耗时对照：
//
// 运行（必须在 bin/ 下跑，配置与视频路径都是相对 bin/ 的）：
//   cd bin && ./test_schedule_strategy                               # 全量策略对照 + 降幅表
//   cd bin && ./test_schedule_strategy --benchmark_filter=Stagger    # 只看错峰
//   cd bin && ./test_schedule_strategy --benchmark_repetitions=3     # 三次运行（样本累加，降幅看中位数）
//

#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"
#include "scope_timer.h"

#include <benchmark/benchmark.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// 预热
constexpr int kWarmupFrames = 20;
// 每个策略统计的帧数
constexpr int kFramesPerRun = 100;

// ------------------------------- 配置与全局对象 -------------------------------
LoggerManager & logger_manager = LoggerManager::getInstance(false, true, "err");

std::string config_path = "benchmark.yaml";
YAML::Node  root        = YAML::LoadFile(config_path);
std::string video_path  = root["video_path"].as<std::string>();

// 模型路径：优先读本测试的专用节点，没有就回落到 test_trt_pipeline 的节点
const std::string k_schedule_task_node = "test_schedule_strategy";

YAML::Node resolveTaskNode() {
    YAML::Node node = root["task"][k_schedule_task_node];
    if (!node || !node["yolo_model_path"]) {
        node = root["task"]["test_trt_pipeline"];
    }
    if (!node || !node["yolo_model_path"] || !node["depth_model_path"]) {
        std::cerr << "[FATAL] " << config_path << " has neither task." << k_schedule_task_node
                  << " nor task.test_trt_pipeline with yolo_model_path / depth_model_path"
                  << std::endl;
        std::exit(1);
    }
    return node;
}

YAML::Node  task_node        = resolveTaskNode();
std::string yolo_model_path  = task_node["yolo_model_path"].as<std::string>();
std::string depth_model_path = task_node["depth_model_path"].as<std::string>();
bool        use_gpu          = task_node["use_gpu"].as<bool>(true);

IOManager io_manager("none");
FrameMeta frame_meta = io_manager.Init(video_path);
Pipeline  pipeline(depth_model_path, yolo_model_path, frame_meta, use_gpu);

// ------------------------------- 策略与统计 ----------------------------------
struct StrategySpec {
    std::string  label;            // benchmark 名 / 统计表 key
    ScheduleMode mode;             // 调度策略
    int          detect_interval;  // 仅 STAGGER 生效：N = 每 N 帧推一次检测
    int          depth_interval;   // 仅 STAGGER 生效：N = 每 N 帧推一次深度
};

struct StrategyStats {
    StrategySpec spec;
    std::vector<double> frame_us;  // 全部样本帧的 process() 耗时（µs，与 main.cpp 的埋点同单位）
    std::vector<double> dual_path_us;  // 其中检测与深度同帧（双路帧）的样本
    long                detect_frames    = 0;
    long                depth_frames     = 0;
    long                dual_path_frames = 0;
};

std::map<std::string, StrategyStats> schedule_stats;

// 与 Pipeline::shouldRunDetect / shouldRunDepth 同一套判据（frame_id % interval == 0），
// 记下来是为了在表里解释"这个策略为什么快"：错峰的均摊耗时被单路帧拉下来了
void recordFrameSample(const StrategySpec & spec, long frame_id, double elapsed_us) {
    StrategyStats & stats = schedule_stats[spec.label];
    stats.spec            = spec;
    stats.frame_us.push_back(elapsed_us);

    const bool is_stagger   = spec.mode == ScheduleMode::STAGGER;
    const bool detect_frame = !is_stagger || (frame_id % spec.detect_interval == 0);
    const bool depth_frame  = !is_stagger || (frame_id % spec.depth_interval == 0);

    if (detect_frame) {
        ++stats.detect_frames;
    }
    if (depth_frame) {
        ++stats.depth_frames;
    }
    if (detect_frame && depth_frame) {
        ++stats.dual_path_frames;
        stats.dual_path_us.push_back(elapsed_us);
    }
}

// 跑一组策略：覆盖调度与间隔 -> 预热 -> 逐帧计时（取流不计入）
void runStrategy(benchmark::State & state, const StrategySpec & spec) {
    pipeline.setScheduleMode(spec.mode);
    pipeline.setStaggerIntervals(spec.detect_interval, spec.depth_interval);

    FrameInputContext  frame_input_context(0, frame_meta);
    InferOutputContext infer_output_context;

    // 用本策略预热，再把取流位置退回第 0 帧，让统计区间从 frame_id = 0 开始，
    // 三个策略的取模相位完全一致（错峰的调度相位本身就是被测对象的一部分）
    for (int i = 0; i < kWarmupFrames; ++i) {
        frame_input_context.setFrameID(i);
        if (!io_manager.readNextFrame(frame_input_context, false) ||
            frame_input_context.raw_img.empty()) {
            break;
        }
        pipeline.process(frame_input_context, infer_output_context);
    }
    io_manager.Init(video_path);

    long frame_id = 0;
    for (auto _ : state) {
        state.PauseTiming();  // 取流（含 H2D）不列入单帧推理耗时
        if (!io_manager.readNextFrame(frame_input_context, false) ||
            frame_input_context.raw_img.empty()) {
            io_manager.Init(video_path);  // 视频播完回到开头，保证样本数 = iterations
            continue;
        }
        frame_input_context.setFrameID(frame_id);
        state.ResumeTiming();

        const auto begin = std::chrono::steady_clock::now();
        pipeline.process(frame_input_context, infer_output_context);
        const auto end = std::chrono::steady_clock::now();

        state.PauseTiming();
        recordFrameSample(spec, frame_id,
                          std::chrono::duration<double, std::micro>(end - begin).count());
        ++frame_id;
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations());
}

class ScheduleBenchmark : public benchmark::Fixture {};

// 串行：两路依次同步执行，作为降幅基线
BENCHMARK_DEFINE_F(ScheduleBenchmark, Sync)(benchmark::State & state) {
    runStrategy(state, { "sync", ScheduleMode::SYNC, 1, 1 });
}

// 重叠：本帧检测与深度异步提交到各自 stream 重叠执行
BENCHMARK_DEFINE_F(ScheduleBenchmark, Overlap)(benchmark::State & state) {
    runStrategy(state, { "overlap", ScheduleMode::OVERLAP, 1, 1 });
}

// 错峰 1×1：每帧都是双路帧 -> 应退化为重叠路径，用来验证"错峰只是调度，不是另一套推理"
BENCHMARK_DEFINE_F(ScheduleBenchmark, Stagger_Detect1_Depth1)(benchmark::State & state) {
    runStrategy(state, { "stagger(1x1)", ScheduleMode::STAGGER, 1, 1 });
}

// 错峰 1×3：实机 config_jetson.yaml 的配置（检测每帧、深度每 3 帧）
BENCHMARK_DEFINE_F(ScheduleBenchmark, Stagger_Detect1_Depth3)(benchmark::State & state) {
    runStrategy(state, { "stagger(1x3)", ScheduleMode::STAGGER, 1, 3 });
}

// 错峰 2×3：碰撞帧只在 6 的倍数帧出现，单路帧占比最高
BENCHMARK_DEFINE_F(ScheduleBenchmark, Stagger_Detect2_Depth3)(benchmark::State & state) {
    runStrategy(state, { "stagger(2x3)", ScheduleMode::STAGGER, 2, 3 });
}

// 错峰 3×3：检测与深度同频，除双路帧外全部是空帧（只跑收尾的运动状态判定），最省算力
BENCHMARK_DEFINE_F(ScheduleBenchmark, Stagger_Detect3_Depth3)(benchmark::State & state) {
    runStrategy(state, { "stagger(3x3)", ScheduleMode::STAGGER, 3, 3 });
}

BENCHMARK_REGISTER_F(ScheduleBenchmark, Sync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(kFramesPerRun)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ScheduleBenchmark, Overlap)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(kFramesPerRun)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ScheduleBenchmark, Stagger_Detect1_Depth1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(kFramesPerRun)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ScheduleBenchmark, Stagger_Detect1_Depth3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(kFramesPerRun)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ScheduleBenchmark, Stagger_Detect2_Depth3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(kFramesPerRun)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ScheduleBenchmark, Stagger_Detect3_Depth3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(kFramesPerRun)
    ->UseRealTime();

// ------------------------------- 结论表 -------------------------------------
// 打印顺序固定（不跟 google benchmark 的字典序），保证每次输出的对照位置一致
const std::vector<std::string> kReportOrder = { "sync",         "overlap",      "stagger(1x1)",
                                                "stagger(1x3)", "stagger(2x3)", "stagger(3x3)" };

void printHeader() {
    std::cout << "\n=== Schedule comparison setup ===\n"
              << "  video        : " << video_path << "\n"
              << "  detect model : " << yolo_model_path << "\n"
              << "  depth model  : " << depth_model_path << "\n"
              << "  frames       : frame 0 .. " << (kFramesPerRun - 1) << " per strategy ("
              << kWarmupFrames << " warmup frames excluded)\n"
              << "  timed region : pipeline.process() only; frame read (incl. H2D), save and draw "
                 "are excluded\n"
              << "  backend      : detect=" << pipeline.getDetector().backendTypeName()
              << " (async=" << (pipeline.getDetector().isAsyncInferenceSupported() ? "yes" : "no")
              << "), depth=" << pipeline.getActiveDepthModel().backendTypeName() << " (async="
              << (pipeline.getActiveDepthModel().isAsyncInferenceSupported() ? "yes" : "no")
              << ")\n"
              << "  note         : stagger collision frames reuse the main detector here (the\n"
              << "                 yolo26n light model is not loaded), so the stagger gain shown\n"
              << "                 below is a LOWER BOUND of what config_jetson.yaml achieves\n";
    if (!pipeline.getDetector().isAsyncInferenceSupported() ||
        !pipeline.getActiveDepthModel().isAsyncInferenceSupported()) {
        std::cout << "  [WARN] backend without async support -> overlap/stagger fall back to sync "
                     "execution, overlap gain not readable\n";
    }
}

constexpr size_t kTableWidth = 119;

// 居中标题：左右补 '=' 到 kTableWidth，避免手写长度和分隔线对不齐
std::string centeredTitle(const std::string & text) {
    if (text.size() + 2 >= kTableWidth) {
        return text;
    }
    const size_t pad_left = (kTableWidth - text.size()) / 2;
    return std::string(pad_left, '=') + text +
           std::string(kTableWidth - text.size() - pad_left, '=');
}

void printSummary() {
    auto         baseline_it  = schedule_stats.find("sync");
    const bool   has_baseline = baseline_it != schedule_stats.end();
    const double baseline_ms  = has_baseline ? calculateAverage(baseline_it->second.frame_us) : 0.0;

    // 表头用纯 ASCII：中文字符在 std::setw 下按字节计宽，和 ASCII 数据行会错位
    const std::string divider(kTableWidth, '-');
    std::cout << "\n" << centeredTitle(" Per-frame pipeline latency by schedule mode ") << "\n";
    std::cout << std::left << std::setw(15) << "mode" << std::right << std::setw(8) << "n"
              << std::setw(10) << "mean_ms" << std::setw(9) << "p50" << std::setw(9) << "p95"
              << std::setw(9) << "p99" << std::setw(10) << "dual_ms" << std::setw(9) << "detect"
              << std::setw(9) << "depth" << std::setw(9) << "dual" << std::setw(22) << "vs_sync"
              << "\n";
    std::cout << divider << "\n";

    std::cout << std::fixed << std::setprecision(2);
    for (const auto & label : kReportOrder) {
        auto it = schedule_stats.find(label);
        if (it == schedule_stats.end()) {
            continue;
        }
        const StrategyStats & stats     = it->second;
        const double          mean      = calculateAverage(stats.frame_us);
        const double          p50       = calculatePercentile(stats.frame_us, 50.0);
        const double          p95       = calculatePercentile(stats.frame_us, 95.0);
        const double          p99       = calculatePercentile(stats.frame_us, 99.0);
        const double          dual_mean = calculateAverage(stats.dual_path_us);

        std::ostringstream delta;
        if (!has_baseline || label == "sync") {
            delta << (has_baseline ? "baseline" : "no sync baseline");
        } else {
            const double saved = mean - baseline_ms;
            delta << std::setprecision(2) << saved << " ms (" << std::setprecision(1)
                  << (saved / baseline_ms * 100.0) << "%)";
        }

        std::cout << std::left << std::setw(15) << label << std::right << std::setprecision(0)
                  << std::setw(8) << static_cast<long>(stats.frame_us.size())
                  << std::setprecision(2) << std::setw(10) << mean << std::setw(9) << p50
                  << std::setw(9) << p95 << std::setw(9) << p99 << std::setw(10) << dual_mean
                  << std::setw(9) << stats.detect_frames << std::setw(9) << stats.depth_frames
                  << std::setw(9) << stats.dual_path_frames << std::setw(22) << delta.str() << "\n";
    }
    std::cout << divider << "\n"
              << "mean/p50/p95/p99 = per-frame wall clock of pipeline.process() in ms\n"
              << "dual_ms = mean of frames running BOTH detect and depth (every frame for\n"
              << "          sync/overlap, collision frames only for stagger)\n"
              << "detect/depth/dual = frames that actually ran detect / depth / both models\n"
              << "vs_sync = delta against serial mode in ms per frame (negative = cheaper)\n"
              << "max inference-only fps = 1000 / mean_ms (excludes frame read and save,\n"
              << "                         so it is NOT the end-to-end frame rate)\n\n";

    // 结论：直接把每档的降幅算出来
    if (has_baseline) {
        std::cout << "Conclusion (vs serial, mean per frame):\n" << std::setprecision(2);
        for (const auto & label : kReportOrder) {
            auto it = schedule_stats.find(label);
            if (it == schedule_stats.end() || label == "sync") {
                continue;
            }
            const double mean  = calculateAverage(it->second.frame_us);
            const double saved = mean - baseline_ms;
            std::cout << "  " << std::left << std::setw(14) << label << std::right << "saves "
                      << -saved << " ms/frame (" << std::setprecision(1)
                      << (-saved / baseline_ms * 100.0) << "%), inference-only fps up to "
                      << std::setprecision(0) << (1000.0 / mean);
            std::cout << std::setprecision(2) << "\n";
        }
        std::cout << "\n";
    }

    if (!has_baseline) {
        std::cout << "sync baseline was not run (filtered out), delta vs serial unavailable\n";
    }
}

int main(int argc, char ** argv) {
    printHeader();

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();

    printSummary();
    return 0;
}
