#pragma once
// 运动/危险判定引擎 + 快速靠近（approach）检测
//
// 本文件分两部分：
//   1) namespace approach：算法本体
//        * SignalFilterBank            框高/深度的因果前置滤波（1€ / 卡尔曼 / 不过滤）
//        * ApproachDetectorCumulative  方案 d 快速靠近判定（基线 + 累计变化 + 双边去抖）
//   2) MotionStateEngine：对外接口（配置 / 逐帧判定 / 框内深度采样）
//
// 接近判定按 track 逐个进行（人/车各自独立，不做任何合并），由调用方负责类别筛选。
//
// 滤波器不自己实现，直接用现成的库：
//   * 1€/one_euro -> 第三方库 casiez/OneEuroFilter（BSD-3-Clause），git submodule 于
//                    third_party/OneEuroFilter，见 CMake 目标 one_euro_filter；
//   * kalman      -> OpenCV cv::KalmanFilter（预测/更新/增益由 OpenCV 负责，这里只装配
//                    状态模型 [值, 速度] 与 F/H/Q/R）；
//   * none        -> 原值透传。
//
// 约定：
//   * 逐帧在线计算，只依赖历史（因果），不做整段轨迹的回看或平滑；
//   * 中位数取“上中位”、只统计 > 0 的样本、score 分母加 1e-3 下限，内部计算用 double；
//   * 时间戳统一为秒（视频 = frame_id / fps；相机 = 系统时钟）；
//   * 阈值默认值与 bin/config.yaml 的 motion_state_engine.approach 段一致（见 ConfigManager）：
//       filter=one_euro, score_thr=0.45, confirm=3, thr_depth=0.15, thr_height=0.20,
//       exit_score_thr=0.2, exit_confirm=3

#include "frame.h"
#include "OneEuroFilter.h"  // third_party/OneEuroFilter/cpp（CMake 目标 one_euro_filter 提供）
#include "STrack.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <opencv2/core/mat.hpp>        // cv::Mat
#include <opencv2/video/tracking.hpp>  // cv::KalmanFilter
#include <string>
#include <unordered_map>
#include <vector>

namespace approach {

// [因果滤波] 1€ / 一维匀速卡尔曼：均为库实现，只依赖历史，供接近检测前置去噪

// 1€ 滤波参数（当前采用的一组：低频截止 1Hz、beta 0.01、微分截止 1Hz）
const double kOneEuroMinCutoff   = 1.0;
const double kOneEuroBeta        = 0.01;
const double kOneEuroDerivCutoff = 1.0;

enum class FilterMode { NONE, ONE_EURO, KALMAN };

// 解析配置字符串："none" / "one_euro" / "kalman"（大小写不敏感）
FilterMode parseFilterMode(const std::string & name);

const char * filterModeName(FilterMode mode);

// 对每条 track 的框高 / 深度各自做因果滤波，供接近检测器前置使用。
//   * one_euro 模式：casiez/OneEuroFilter 库对象，每条 track 的 height/depth 各一个；
//   * kalman 模式：cv::KalmanFilter，2 状态 [值, 速度]，height 在 log 域滤波（更贴合指数增长）；
//   * depth <= 0（无效）时不过滤，原值透传；
//   * mode == NONE 时原值透传，不做任何平滑。
// freq 仅作为“没有时间戳时”的采样频率估计；一旦调用方传入递增时间戳，
// 库会按时间戳自行更新频率，因此默认 30Hz 只影响前两帧。
class SignalFilterBank {
  public:
    explicit SignalFilterBank(FilterMode mode, double freq = 30.0);

    void update(int      track_id,
                double   height,
                double   depth,
                double   ts,
                double & height_filtered,
                double & depth_filtered);

  private:
    // 每条 track 的滤波状态：1€ 用库对象（unique_ptr：库对象持有内部指针、不可拷贝），
    // kalman 用 cv::KalmanFilter + 首帧自动噪声估计所需的状态
    struct TrackFilters {
        explicit TrackFilters(double freq);

        std::unique_ptr<OneEuroFilter> height_one_euro;
        std::unique_ptr<OneEuroFilter> depth_one_euro;

        cv::KalmanFilter height_kalman;  // 2 状态 [log(height), d log(height)/dt]
        cv::KalmanFilter depth_kalman;   // 2 状态 [depth, d depth/dt]
        bool             height_ready  = false;
        bool             depth_ready = false;
        double           height_q      = 0.0;  // 首帧按幅值估计的过程噪声（速度方差）
        double           depth_q     = 0.0;
        double           height_t      = 0.0;  // 上一次时间戳
        double           depth_t     = 0.0;
    };

    // 一维匀速卡尔曼单步（cv::KalmanFilter 负责预测/更新/增益，这里只装配模型与 dt）
    static double kalmanStep(cv::KalmanFilter & kf,
                             bool &             ready,
                             double &           q,
                             double &           t_prev,
                             double             z,
                             double             t,
                             bool               log_domain);

    FilterMode                                             mode_;
    double                                                 freq_;
    std::unordered_map<int, std::unique_ptr<TrackFilters>> tracks_;
};

// [快速靠近检测] 方案 d：基线 + 累计变化率 + 近期趋势（持续判断，进出双边去抖）

struct ApproachParams {
    int    warmup         = 30;    // 基线攒帧数（前 warmup 帧必然无输出）
    double thr_depth      = 0.20;  // 相对基线的深度降幅阈值（0.15 = 15%）
    double thr_height     = 0.30;  // 相对基线的框高增幅阈值（0.20 = 20%）
    int    recent_w       = 10;    // 近期趋势窗口长度（前后半窗各 recent_w/2）
    double score_thr      = 0.60;  // 进入分数线
    int    confirm        = 2;     // 进入需连续达标帧数
    double exit_score_thr = 0.30;  // 退出分数线（<= 它或趋势消失计为退出证据）
    int    exit_confirm   = 3;     // 退出需连续证据帧数
};

struct ApproachState {
    bool  alarm         = false;  // 本帧接近报警（已通过双边迟滞去抖）
    bool  alarm_started = false;  // 本帧为报警上升沿（未报警 -> 报警）
    bool  alarm_cleared = false;  // 本帧为报警下降沿（报警 -> 解除）
    float score         = 0.0f;   // 融合分数 = 0.4 * 深度分 + 0.6 * 尺度分
    float depth_score   = 0.0f;   // 深度分（累计降幅 / thr_depth，截断 0~1）
    float scale_score   = 0.0f;   // 尺度分（框高累计增幅 / thr_height，截断 0~1）
    // 注：三项分数为“最近一次有效计算值”：
    //     本帧窗口内无有效样本（如深度被滤波过冲到 <= 0）时保持上一帧分数，不归零。
};

// 逐帧在线判定“目标是否正在快速靠近”。
//
// 原理：先用轨迹前 warmup 帧攒基线（深度/框高的中位数，抗噪），之后每帧基于基线算累计变化：
//         depth_drop  = (基线深度 - 当前深度) / 基线深度
//         height_gain = 当前框高 / 基线框高 - 1
//       并加“近期仍在靠近”的门控（最近 recent_w 帧内：深度后半窗 < 前半窗 且框高后半窗 >
//       前半窗），保证只报“当前正在靠近”，而不是“曾经靠近过”。
//
// 进出判定（双边迟滞 + 去抖）：
//       进入证据：趋势在靠近 且 score >= score_thr，连续 confirm 帧 -> 报警
//       退出证据：趋势不再靠近 或 score <= exit_score_thr，连续 exit_confirm 帧 -> 解除
//       中间态（分数介于两线之间且趋势未消失）：既不进也不退，保持当前状态，抗闪烁。
//
// 优点：中位数基线抗噪；后期才开始的靠近也能报；退出也要连续帧确认，单帧抖动不会让报警闪烁。
// 缺点：前 warmup 帧必然无输出；基线质量依赖前 warmup 帧的稳定程度。
class ApproachDetectorCumulative {
  public:
    explicit ApproachDetectorCumulative(const ApproachParams & params = ApproachParams());

    // 逐帧更新。height = 目标框高（像素），depth = 框内有效深度（> 0 才有效），ts 单位秒
    ApproachState update(int track_id, double height, double depth, double ts);

    const ApproachParams & params() const { return params_; }

    // 热更新（控制面板滑动条）：只改阈值/去抖帧数/窗口长度，不清空已累积的轨迹状态。
    // 注意 warmup 改动只对之后新建立的轨迹（新 track id）生效。
    void setWarmup(int value);
    void setThrDepth(double value);
    void setThrHeight(double value);
    void setRecentW(int value);
    void setScoreThr(double value);
    void setConfirm(int value);
    void setExitScoreThr(double value);
    void setExitConfirm(int value);

  private:
    struct Sample {
        double height = 0.0;
        double depth  = 0.0;
    };

    struct TrackState {
        std::vector<Sample> hist;  // 攒基线阶段的样本缓存
        bool                has_baseline = false;
        double              baseline_d   = 0.0;
        double              baseline_h   = 0.0;
        std::deque<Sample>  recent;           // 最近 recent_w 帧（height, depth）
        int                 streak      = 0;  // 连续进入证据帧数
        int                 exit_streak = 0;  // 连续退出证据帧数
        bool                alarm       = false;
        double              score       = 0.0;
        double              depth_score = 0.0;
        double              scale_score = 0.0;
    };

    // 中位数：先剔除 <= 0 / 非有限样本，排序后取上中位（len/2）；样本全无效返回 false
    static bool medianPositive(const std::vector<double> & values, double & out);

    static double clip01(double value);

    // 把“最近一次有效计算”的三项分数写入输出（无有效窗口时保持上一帧值）
    static void applyScores(const TrackState & state, ApproachState & out);

    ApproachParams                      params_;
    std::unordered_map<int, TrackState> tracks_;
};

}  // namespace approach

// [MotionStateEngine] 对外接口

// 运动/危险判定引擎：只保留“快速靠近”（approach）一路（算法本体见上面的 namespace approach）：
//   先用每条轨迹前 warmup 帧攒基线（深度/框高中位数），之后每帧算相对基线的累计变化率，
//   并要求最近 recent_w 帧“仍在靠近”（深度降 + 框高涨）；进入报警需连续 confirm 帧达标，
//   解除需连续 exit_confirm 帧出现退出证据（双边迟滞去抖）。
// 判定逐 track 独立进行：人/车各自用自己的框高 + 框内深度，不做任何合并；
// 哪些类别参与判定由调用方筛选（pipeline 只送 bicycle/car/motorcycle/bus/truck）。
// 结果写入 MotionStateInfoRecord::approach_*，快速靠近即视为危险目标（报警 + 画红框）。
//
// 说明：原先基于尺度/深度卡尔曼速度的“运动状态 + TTC”一路（含其配置项与控制面板滑动条）
// 已按需求整体移除，危险判定不再使用 TTC。
class MotionStateEngine {
  public:
    MotionStateEngine() = default;

    // 框内鲁棒深度：缩进 20% 边界后网格采样，剔除两端 25% 后取中间 50% 的均值（截断均值）
    float computeMeanDepth(cv::Mat                    depth,
                           const std::vector<float> & tlwh,
                           int                        num_samples = 64) const;

  private:
    // 快速靠近检测（方案 d 检测器 + 可选因果滤波）
    // 默认关闭：只有 Pipeline 按 config 调用 configureApproach() 后才生效，
    // 保证 benchmark 等未配置的调用点行为不变
    bool                                        approach_enabled_ = false;
    approach::ApproachDetectorCumulative        approach_detector_;
    std::unique_ptr<approach::SignalFilterBank> approach_filter_bank_;

  public:
    // 快速靠近检测（方案 d）
    // 配置一次即可（Pipeline 构造后按 config 调用）。filter_mode 为 none/one_euro/kalman，
    // enabled=false 时整条路关闭（updateApproachState 直接返回未报警）。
    void configureApproach(const approach::ApproachParams & params,
                           approach::FilterMode            filter_mode,
                           bool                            enabled);

    // 逐帧更新某个 track 的判定。内部先按配置做 height/depth 因果滤波（若启用），
    // 再跑方案 d 检测器；返回本帧的报警/分数（分数为最近一次有效计算值）。
    //   height    : 该 track 的框高（像素），来自 tlwh
    //   raw_depth : 该 track 框内深度（computeMeanDepth；<= 0 表示无效，会原值透传不滤波）
    //   timestamp : 秒（视频 = frame_id / fps；相机 = 系统时钟）
    approach::ApproachState updateApproachState(int    track_id,
                                                float  height,
                                                float  raw_depth,
                                                double timestamp);

    // 参数读写（阈值可运行时热更新，供控制面板滑动条使用；warmup 只影响之后新建的轨迹）
    double getApproachThrDepth() const { return approach_detector_.params().thr_depth; }

    void setApproachThrDepth(double value) { approach_detector_.setThrDepth(value); }

    double getApproachThrHeight() const { return approach_detector_.params().thr_height; }

    void setApproachThrHeight(double value) { approach_detector_.setThrHeight(value); }

    double getApproachScoreThr() const { return approach_detector_.params().score_thr; }

    void setApproachScoreThr(double value) { approach_detector_.setScoreThr(value); }

    int getApproachConfirm() const { return approach_detector_.params().confirm; }

    void setApproachConfirm(int value) { approach_detector_.setConfirm(value); }

    double getApproachExitScoreThr() const { return approach_detector_.params().exit_score_thr; }

    void setApproachExitScoreThr(double value) { approach_detector_.setExitScoreThr(value); }

    int getApproachExitConfirm() const { return approach_detector_.params().exit_confirm; }

    void setApproachExitConfirm(int value) { approach_detector_.setExitConfirm(value); }
};
