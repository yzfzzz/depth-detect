#pragma once
// 快速靠近（approach）检测：逐 track 判定目标是否正在快速靠近，报警即危险依据，
// 结果写入 MotionStateInfoRecord::approach_*。
// 设计约束（why）：
//   * 全程因果在线、只依赖历史，不做整段轨迹回看——逐帧实时触发的要求；
//   * 滤波：1€ 或 Kalman，滤波器状态按 track_id 维护；滤波器仅在 approach_enabled_ 时创建，避免无用开销；
//   * 中位数基线、只统计 > 0 的样本、score 分母 1e-3 下限：抗单帧噪声，且阈值配 0
//   * 时间戳统一为秒（视频 = frame_id / fps；相机 = 系统时钟）；

#include "OneEuroFilter.h"  // third_party/OneEuroFilter/cpp（CMake 目标 one_euro_filter 提供）
#include <cmath>
#include <deque>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace approach {

// 经验调优值，未随 config 暴露
const double kOneEuroMinCutoff   = 1.0;
const double kOneEuroBeta        = 0.01;
const double kOneEuroDerivCutoff = 1.0;

enum class FilterMode { NONE, ONE_EURO, KALMAN };

FilterMode parseFilterMode(const std::string & name);

const char * filterModeName(FilterMode mode);

// 对每条 track 的框高/深度做前置因果滤波。
// 框高在 log 域滤波：目标靠近时框高近指数增长，线性域滤波会系统性滞后；
// 深度 <= 0 视为无效、原值透传——0 值会把滤波器带偏。
// freq 仅在调用方不传时间戳时充作采样频率估计，之后由库按时间戳自适应。
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
    struct TrackFilters {
        explicit TrackFilters(double freq);

        std::unique_ptr<OneEuroFilter> height_one_euro;
        std::unique_ptr<OneEuroFilter> depth_one_euro;

        cv::KalmanFilter height_kalman;  // 2 状态 [log(height), d log(height)/dt]
        cv::KalmanFilter depth_kalman;   // 2 状态 [depth, d depth/dt]
        bool             height_ready = false;
        bool             depth_ready  = false;
        double height_q      = 0.0;  // 过程噪声（速度方差），首帧按量测幅值估计
        double depth_q       = 0.0;
        double height_t_prev = 0.0;
        double depth_t_prev  = 0.0;
    };

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

struct ApproachParams {
    int    warmup     = 30;    // 基线帧数；此期间无任何输出
    double thr_depth  = 0.20;  // 深度相对基线的降幅阈值（0.2 = 20%），无量纲比值
    double thr_height = 0.30;  // 框高相对基线的增幅阈值（0.2 = 20%）
    int    recent_w   = 10;    // 近期趋势窗口长度（前后半窗各 recent_w/2）
    double score_thr  = 0.60;  // 进入分数线
    int    confirm    = 2;     // 连续达标帧数才进报警
    double exit_score_thr = 0.30;  // 退出分数线
    int    exit_confirm   = 3;     // 连续退出证据帧数才解除
};

struct ApproachState {
    bool  alarm         = false;  // 已过双边去抖的最终判定
    bool  alarm_started = false;  // 报警上升沿（未报警 -> 报警）
    bool  alarm_cleared = false;  // 报警下降沿（报警 -> 解除）
    float score         = 0.0f;
    float depth_score   = 0.0f;
    float scale_score   = 0.0f;  // 基于框高变化
    // 三项分数为最近一次有效计算的值：无有效窗口的帧保持上帧值，不归零
};

// 判定"目标是否正在快速靠近"。
//
// 基线取前 warmup 帧的中位数（抗单帧噪声）；此后每帧算相对基线的累计变化，
// 并要求最近 recent_w 帧仍在靠近（深度降 + 框高涨）——保证只报"正在靠近"，
// 而不是"曾经靠近过"的目标持续报警。
// 进/出都要求连续帧证据（双边迟滞）：单帧分数抖动不会让报警闪烁；
// 代价是 warmup 期间无输出，基线质量取决于该时段目标是否稳定。
class ApproachDetectorCumulative {
  public:
    explicit ApproachDetectorCumulative(const ApproachParams & params = ApproachParams());

    // height = 框高（像素）；depth <= 0 视为无效
    ApproachState update(int track_id, double height, double depth, double ts);

    const ApproachParams & params() const { return params_; }

    // 热更新（控制面板滑动条）：不清空已累积的轨迹状态；warmup 只对新轨迹生效
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
        std::vector<Sample> hist;
        bool                has_baseline = false;
        double              baseline_d   = 0.0;
        double              baseline_h   = 0.0;
        std::deque<Sample>  recent;
        int                 streak      = 0;  // 连续进入证据帧数
        int                 exit_streak = 0;  // 连续退出证据帧数
        bool                alarm       = false;
        double              score       = 0.0;
        double              depth_score = 0.0;
        double              scale_score = 0.0;
    };

    // 只统计 > 0 且有限的样本，全无效返回 false
    static bool medianPositive(const std::vector<double> & values, double & out);

    static double clip01(double value);

    static void applyScores(const TrackState & state, ApproachState & out);

    ApproachParams                      params_;
    std::unordered_map<int, TrackState> tracks_;
};

}  // namespace approach

// 对外门面：配置 + 逐帧判定 + 框内深度采样，算法本体在 namespace approach。
// 判定逐 track 独立（人/车不合并），参与判定的类别由调用方筛选。
class MotionStateEngine {
  public:
    MotionStateEngine() = default;

    // 框内鲁棒深度：缩进边界避开框缘混入的背景，截断均值剔除前后各 25% 异常值
    float computeMeanDepth(cv::Mat                    depth,
                           const std::vector<float> & tlwh,
                           int                        num_samples = 64) const;

  private:
    // 默认关闭：未 configureApproach 的调用点（如 benchmark）行为保持不变
    bool                                        approach_enabled_ = false;
    approach::ApproachDetectorCumulative        approach_detector_;
    std::unique_ptr<approach::SignalFilterBank> approach_filter_bank_;

  public:
    // Pipeline 构造后调用一次即可；enabled=false 时 updateApproachState 恒返回未报警
    void configureApproach(const approach::ApproachParams & params,
                           approach::FilterMode             filter_mode,
                           bool                             enabled);

    // raw_depth <= 0 视为无效（原值透传不滤波）；timestamp 单位秒
    approach::ApproachState updateApproachState(int    track_id,
                                                float  height,
                                                float  raw_depth,
                                                double timestamp);

    // 参数热更新（控制面板滑动条）
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
