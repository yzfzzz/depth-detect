#include "motion_state_engine.h"

#include "logger_manager.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace approach {

namespace {

const double kMinDt = 1e-6;

// log 域下限，防 log(0)
const double kMinLogValue = 1e-6;

// score 分母下限：阈值配 0 时 score 退化为 0/1 跳变，但不产生 inf/NaN
const double kMinThreshold = 1e-3;

cv::Mat mat2x2(double a00, double a01, double a10, double a11) {
    cv::Mat m          = cv::Mat::zeros(2, 2, CV_64F);
    m.at<double>(0, 0) = a00;
    m.at<double>(0, 1) = a01;
    m.at<double>(1, 0) = a10;
    m.at<double>(1, 1) = a11;
    return m;
}

cv::Mat mat1x2(double a, double b) {
    cv::Mat m          = cv::Mat::zeros(1, 2, CV_64F);
    m.at<double>(0, 0) = a;
    m.at<double>(0, 1) = b;
    return m;
}

cv::Mat mat1x1(double value) {
    cv::Mat m          = cv::Mat::zeros(1, 1, CV_64F);
    m.at<double>(0, 0) = value;
    return m;
}

}  // namespace

FilterMode parseFilterMode(const std::string & name) {
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "one_euro" || lower == "oneeuro" || lower == "one-euro" || lower == "euro") {
        return FilterMode::ONE_EURO;
    }
    if (lower == "kalman" || lower == "kf") {
        return FilterMode::KALMAN;
    }
    return FilterMode::NONE;
}

const char * filterModeName(FilterMode mode) {
    switch (mode) {
        case FilterMode::ONE_EURO:
            return "one_euro";
        case FilterMode::KALMAN:
            return "kalman";
        case FilterMode::NONE:
        default:
            return "none";
    }
}

SignalFilterBank::TrackFilters::TrackFilters(double freq) {
    height_one_euro.reset(
        new OneEuroFilter(freq, kOneEuroMinCutoff, kOneEuroBeta, kOneEuroDerivCutoff));
    depth_one_euro.reset(
        new OneEuroFilter(freq, kOneEuroMinCutoff, kOneEuroBeta, kOneEuroDerivCutoff));
}

SignalFilterBank::SignalFilterBank(FilterMode mode, double freq) : mode_(mode), freq_(freq) {}

double SignalFilterBank::kalmanStep(cv::KalmanFilter & kf,
                                    bool &             ready,
                                    double &           q,
                                    double &           t_prev,
                                    double             z,
                                    double             t,
                                    bool               log_domain) {
    if (log_domain) {
        z = std::log(std::max(z, kMinLogValue));
    }

    // 首帧无历史可预测，直接用量测初始化并原值返回；
    // R/Q 按量测幅值自动估计，避免固定噪声参数在近/远目标上失配
    if (!ready) {
        ready = true;

        kf.init(2, 1, 0, CV_64F);
        kf.measurementMatrix          = mat1x2(1.0, 0.0);
        const double r                = std::pow(0.05 * std::abs(z) + 1e-3, 2);
        q                             = std::pow(0.03 * std::abs(z) + 1e-3, 2);
        kf.measurementNoiseCov        = mat1x1(r);
        kf.errorCovPost               = mat2x2(r, 0.0, 0.0, q);
        kf.statePost                  = cv::Mat::zeros(2, 1, CV_64F);
        kf.statePost.at<double>(0, 0) = z;
        t_prev                        = t;
        return log_domain ? std::exp(z) : z;
    }

    const double dt = std::max(t - t_prev, kMinDt);
    t_prev          = t;

    // 匀速模型的 F 与过程噪声离散化（由速度噪声 q 与 dt 导出）
    kf.transitionMatrix = mat2x2(1.0, dt, 0.0, 1.0);
    const double dt2    = dt * dt;
    kf.processNoiseCov =
        mat2x2(q * dt2 * dt2 / 4.0, q * dt2 * dt / 2.0, q * dt2 * dt / 2.0, q * dt2);

    kf.predict();
    kf.correct(mat1x1(z));
    return kf.statePost.at<double>(0, 0);
}

void SignalFilterBank::update(int      track_id,
                              double   height,
                              double   depth,
                              double   ts,
                              double & height_filtered,
                              double & depth_filtered) {
    if (mode_ == FilterMode::NONE) {
        height_filtered = height;
        depth_filtered  = depth;
        return;
    }

    std::unique_ptr<TrackFilters> & holder = tracks_[track_id];
    if (!holder) {
        holder.reset(new TrackFilters(freq_));
    }
    TrackFilters & filters = *holder;

    if (mode_ == FilterMode::ONE_EURO) {
        height_filtered = filters.height_one_euro->filter(height, ts);
    } else {
        height_filtered = kalmanStep(filters.height_kalman, filters.height_ready, filters.height_q,
                                     filters.height_t_prev, height, ts, true);
    }

    if (depth > 0.0) {
        if (mode_ == FilterMode::ONE_EURO) {
            depth_filtered = filters.depth_one_euro->filter(depth, ts);
        } else {
            depth_filtered = kalmanStep(filters.depth_kalman, filters.depth_ready, filters.depth_q,
                                        filters.depth_t_prev, depth, ts, false);
        }
    } else {
        depth_filtered = depth;
    }
}

ApproachDetectorCumulative::ApproachDetectorCumulative(const ApproachParams & params) :
    params_(params) {
    // 走 setter 复用参数钳制（warmup >= 1、recent_w >= 2 等）
    setWarmup(params.warmup);
    setThrDepth(params.thr_depth);
    setThrHeight(params.thr_height);
    setRecentW(params.recent_w);
    setScoreThr(params.score_thr);
    setConfirm(params.confirm);
    setExitScoreThr(params.exit_score_thr);
    setExitConfirm(params.exit_confirm);
}

void ApproachDetectorCumulative::setWarmup(int value) {
    params_.warmup = std::max(1, value);
}

void ApproachDetectorCumulative::setThrDepth(double value) {
    params_.thr_depth = value;
}

void ApproachDetectorCumulative::setThrHeight(double value) {
    params_.thr_height = value;
}

void ApproachDetectorCumulative::setRecentW(int value) {
    params_.recent_w = std::max(2, value);
}

void ApproachDetectorCumulative::setScoreThr(double value) {
    params_.score_thr = std::max(0.0, std::min(1.0, value));
}

void ApproachDetectorCumulative::setConfirm(int value) {
    params_.confirm = std::max(1, value);
}

void ApproachDetectorCumulative::setExitScoreThr(double value) {
    params_.exit_score_thr = std::max(0.0, std::min(1.0, value));
}

void ApproachDetectorCumulative::setExitConfirm(int value) {
    params_.exit_confirm = std::max(1, value);
}

bool ApproachDetectorCumulative::medianPositive(const std::vector<double> & values, double & out) {
    std::vector<double> xs;
    xs.reserve(values.size());
    for (double value : values) {
        if (std::isfinite(value) && value > 0.0) {
            xs.push_back(value);
        }
    }
    if (xs.empty()) {
        return false;
    }
    std::sort(xs.begin(), xs.end());
    out = xs[xs.size() / 2];
    return true;
}

double ApproachDetectorCumulative::clip01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

ApproachState ApproachDetectorCumulative::update(int    track_id,
                                                 double height,
                                                 double depth,
                                                 double ts) {
    // 判定按帧序推进；ts 只有前置滤波层用，保留参数只为接口一致
    static_cast<void>(ts);

    ApproachState out;
    TrackState &  state = tracks_[track_id];

    // 基线未就绪前只攒样本，不产生判定
    if (!state.has_baseline) {
        state.hist.push_back(Sample{ height, depth });
        if (static_cast<int>(state.hist.size()) < params_.warmup) {
            applyScores(state, out);
            return out;
        }

        std::vector<double> heights;
        std::vector<double> depths;
        heights.reserve(state.hist.size());
        depths.reserve(state.hist.size());
        for (const Sample & sample : state.hist) {
            heights.push_back(sample.height);
            depths.push_back(sample.depth);
        }
        state.hist.clear();

        // 任一通道取不到有效中位数则维持无基线，下一帧重新攒（避免用 0 基线除零）
        if (!medianPositive(depths, state.baseline_d) ||
            !medianPositive(heights, state.baseline_h)) {
            applyScores(state, out);
            return out;
        }
        state.has_baseline = true;
    }

    state.recent.push_back(Sample{ height, depth });
    while (static_cast<int>(state.recent.size()) > params_.recent_w) {
        state.recent.pop_front();
    }
    if (static_cast<int>(state.recent.size()) < params_.recent_w) {
        applyScores(state, out);
        return out;
    }

    const int           half = params_.recent_w / 2;
    std::vector<double> d_prev;
    std::vector<double> d_cur;
    std::vector<double> h_prev;
    std::vector<double> h_cur;
    for (int i = 0; i < params_.recent_w; ++i) {
        const Sample & sample = state.recent[static_cast<size_t>(i)];
        if (i < half) {
            d_prev.push_back(sample.depth);
            h_prev.push_back(sample.height);
        } else {
            d_cur.push_back(sample.depth);
            h_cur.push_back(sample.height);
        }
    }

    double d_prev_median = 0.0;
    double d_cur_median  = 0.0;
    double h_prev_median = 0.0;
    double h_cur_median  = 0.0;
    if (!medianPositive(d_prev, d_prev_median) || !medianPositive(d_cur, d_cur_median) ||
        !medianPositive(h_prev, h_prev_median) || !medianPositive(h_cur, h_cur_median)) {
        // 窗口内出现无效样本（如深度被滤波过冲到 <= 0）：保持上一帧分数
        applyScores(state, out);
        return out;
    }

    const bool trend_ok = d_cur_median < d_prev_median && h_cur_median > h_prev_median;

    const double depth_drop  = (state.baseline_d - d_cur_median) / state.baseline_d;
    const double height_gain = h_cur_median / state.baseline_h - 1.0;
    const double depth_score = clip01(depth_drop / std::max(params_.thr_depth, kMinThreshold));
    const double scale_score = clip01(height_gain / std::max(params_.thr_height, kMinThreshold));
    const double score       = 0.4 * depth_score + 0.6 * scale_score;

    state.score       = score;
    state.depth_score = depth_score;
    state.scale_score = scale_score;

    // 双边迟滞：进/出都需连续帧证据，中间态保持现状，防报警闪烁
    const bool enter_ev = trend_ok && score >= params_.score_thr;
    const bool exit_ev  = !trend_ok || score <= params_.exit_score_thr;

    const bool was_alarm = state.alarm;
    if (state.alarm) {
        if (exit_ev) {
            state.exit_streak += 1;
            if (state.exit_streak >= params_.exit_confirm) {
                state.alarm       = false;
                state.exit_streak = 0;
            }
        } else {
            state.exit_streak = 0;
        }
        state.streak = enter_ev ? (state.streak + 1) : 0;
    } else {
        state.exit_streak = 0;
        if (enter_ev) {
            state.streak += 1;
            if (state.streak >= params_.confirm) {
                state.alarm  = true;
                state.streak = 0;
            }
        } else {
            state.streak = 0;
        }
    }

    out.alarm         = state.alarm;
    out.alarm_started = !was_alarm && state.alarm;
    out.alarm_cleared = was_alarm && !state.alarm;
    applyScores(state, out);
    return out;
}

void ApproachDetectorCumulative::applyScores(const TrackState & state, ApproachState & out) {
    // 保持上一帧分数而非归零：无有效窗口的帧不让下游画面/日志上的分数闪烁
    out.score       = static_cast<float>(state.score);
    out.depth_score = static_cast<float>(state.depth_score);
    out.scale_score = static_cast<float>(state.scale_score);
}

}  // namespace approach

void MotionStateEngine::configureApproach(const approach::ApproachParams & params,
                                          approach::FilterMode             filter_mode,
                                          bool                             enabled) {
    approach_enabled_  = enabled;
    approach_detector_ = approach::ApproachDetectorCumulative(params);
    approach_filter_bank_.reset();

    if (!approach_enabled_) {
        APP_INFO("[Approach] disabled");
        return;
    }
    if (filter_mode != approach::FilterMode::NONE) {
        approach_filter_bank_.reset(new approach::SignalFilterBank(filter_mode));
    }
    APP_INFO(
        "[Approach] enabled: filter={}, warmup={}, thr_depth={:.3f}, thr_height={:.3f}, "
        "recent_w={}, score_thr={:.3f}, confirm={}, exit_score_thr={:.3f}, exit_confirm={}",
        approach::filterModeName(filter_mode), params.warmup, params.thr_depth, params.thr_height,
        params.recent_w, params.score_thr, params.confirm, params.exit_score_thr,
        params.exit_confirm);
}

approach::ApproachState MotionStateEngine::updateApproachState(int    track_id,
                                                               float  height,
                                                               float  raw_depth,
                                                               double timestamp) {
    approach::ApproachState state;
    if (!approach_enabled_) {
        return state;
    }

    double height_filtered = height;
    double depth_filtered  = raw_depth;
    if (approach_filter_bank_) {
        approach_filter_bank_->update(track_id, height, raw_depth, timestamp, height_filtered,
                                      depth_filtered);
    }

    state = approach_detector_.update(track_id, height_filtered, depth_filtered, timestamp);

    // 只在报警沿打日志，避免逐帧刷屏
    if (state.alarm_started) {
        APP_INFO(
            "[Approach] track {} alarm triggered: score={:.3f} depth_score={:.3f} "
            "scale_score={:.3f}",
            track_id, state.score, state.depth_score, state.scale_score);
    } else if (state.alarm_cleared) {
        APP_INFO("[Approach] track {} alarm cleared: score={:.3f}", track_id, state.score);
    }
    return state;
}

float MotionStateEngine::computeMeanDepth(cv::Mat                    depth,
                                          const std::vector<float> & tlwh,
                                          int                        num_samples) const {
    int left   = static_cast<int>(tlwh[0]);
    int top    = static_cast<int>(tlwh[1]);
    int right  = static_cast<int>(tlwh[0] + tlwh[2]);
    int bottom = static_cast<int>(tlwh[1] + tlwh[3]);

    left   = std::max(0, std::min(left, depth.cols - 1));
    top    = std::max(0, std::min(top, depth.rows - 1));
    right  = std::max(0, std::min(right, depth.cols - 1));
    bottom = std::max(0, std::min(bottom, depth.rows - 1));

    int width  = right - left;
    int height = bottom - top;

    if (width <= 0 || height <= 0) {
        return 0.0f;
    }

    std::vector<float> sampled_depths;

    // 缩进 20% 边界：框边缘容易混入背景像素
    int   grid_size    = static_cast<int>(std::sqrt(num_samples));
    float shrink_ratio = 0.2f;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int x = left + static_cast<int>(width * shrink_ratio) +
                    static_cast<int>(i * (width * (1.0f - 2 * shrink_ratio)) /
                                     std::max(1, grid_size - 1));
            int y = top + static_cast<int>(height * shrink_ratio) +
                    static_cast<int>(j * (height * (1.0f - 2 * shrink_ratio)) /
                                     std::max(1, grid_size - 1));

            if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows) {
                continue;
            }

            float depth_value = 0.0f;
            if (depth.type() == CV_32FC1) {
                depth_value = depth.at<float>(y, x);
            } else if (depth.type() == CV_8UC1) {
                depth_value = static_cast<float>(depth.at<uchar>(y, x));
            }

            if (depth_value > 0.0f) {
                sampled_depths.push_back(depth_value);
            }
        }
    }

    if (sampled_depths.empty()) {
        return 0.0f;
    }

    // 截断均值：最近的 25% 多为前景遮挡毛刺、最远的 25% 多为背景噪声
    std::sort(sampled_depths.begin(), sampled_depths.end());
    int num_valid = sampled_depths.size();
    if (num_valid < 4) {
        return sampled_depths[num_valid / 2];  // 样本太少不截断，直接取中位
    }

    int skip_low  = static_cast<int>(num_valid * 0.25f);
    int skip_high = static_cast<int>(num_valid * 0.25f);

    float sum   = 0.0f;
    int   count = 0;
    for (int i = skip_low; i < num_valid - skip_high; ++i) {
        sum += sampled_depths[i];
        count++;
    }

    return count > 0 ? (sum / count) : 0.0f;
}
