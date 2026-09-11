#include "motion_state_engine.h"

#include "logger_manager.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace approach {

namespace {

// 时间步下限（避免 dt 为 0）
const double kMinDt = 1e-6;

// 非负信号在 log 域滤波时的下限（避免 log(0)）
const double kMinLogValue = 1e-6;

// score 分母下限：阈值配 0 时用极小量兜底，
// 保证 thr_depth / thr_height 配 0 时 score 只会在 0/1 之间跳变，不会产生 inf/NaN
const double kMinThreshold = 1e-3;

bool containsId(const std::vector<int> & ids, int class_id) {
    return std::find(ids.begin(), ids.end(), class_id) != ids.end();
}

// 二维小矩阵装配工具（cv::KalmanFilter 负责预测/更新/增益，这里只填 F/H/Q/R 等矩阵）
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

// [SignalFilterBank] 1€ 用第三方库（third_party/OneEuroFilter），kalman 用 OpenCV

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
    // 1€ 库对象：freq 只作为“无时间戳时”的频率估计，实际按传入时间戳自适应更新
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

    // 首帧：按量测幅值自动估计量测/过程噪声并初始化状态，
    // 直接返回原值（无历史可预测）。H = [1, 0]：只观测位置。
    if (!ready) {
        ready = true;

        kf.init(2, 1, 0, CV_64F);
        kf.measurementMatrix          = mat1x2(1.0, 0.0);  // H：只观测位置
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

    // F = [[1, dt], [0, 1]]：匀速模型；Q 由速度过程噪声 q 生成
    kf.transitionMatrix = mat2x2(1.0, dt, 0.0, 1.0);
    const double dt2    = dt * dt;
    kf.processNoiseCov =
        mat2x2(q * dt2 * dt2 / 4.0, q * dt2 * dt / 2.0, q * dt2 * dt / 2.0, q * dt2);

    // 预测 + 更新（增益/协方差更新由 OpenCV 完成），量测为当前帧原始值
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

    // 每条 track 一份滤波器（unique_ptr：库对象持有内部指针，不能拷贝）
    std::unique_ptr<TrackFilters> & holder = tracks_[track_id];
    if (!holder) {
        holder.reset(new TrackFilters(freq_));
    }
    TrackFilters & filters = *holder;

    if (mode_ == FilterMode::ONE_EURO) {
        height_filtered = filters.height_one_euro->filter(height, ts);
    } else {
        // height 非负且接近指数增长 -> 在 log 域滤波（更贴合指数增长）
        height_filtered = kalmanStep(filters.height_kalman, filters.height_ready,
                                     filters.height_q, filters.height_t, height, ts, true);
    }

    // 无效深度（<= 0）不过滤，原值透传，避免 0 值把滤波器带偏
    if (depth > 0.0) {
        if (mode_ == FilterMode::ONE_EURO) {
            depth_filtered = filters.depth_one_euro->filter(depth, ts);
        } else {
            depth_filtered = kalmanStep(filters.depth_kalman, filters.depth_ready, filters.depth_q,
                                        filters.depth_t, depth, ts, false);
        }
    } else {
        depth_filtered = depth;
    }
}

// ApproachDetectorCumulative

ApproachDetectorCumulative::ApproachDetectorCumulative(const ApproachParams & params) :
    params_(params) {
    // 统一走 setter，保证参数合法（warmup >= 1、recent_w >= 2 等）
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
    out = xs[xs.size() / 2];  // 上中位
    return true;
}

double ApproachDetectorCumulative::clip01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

ApproachState ApproachDetectorCumulative::update(int    track_id,
                                                 double height,
                                                 double depth,
                                                 double ts) {
    // 方案 d 只按帧序推进判定；ts 仅由前置滤波层使用（这里保留参数只为接口一致）
    static_cast<void>(ts);

    ApproachState out;
    TrackState &  state = tracks_[track_id];

    // 阶段 1：攒基线（前 warmup 帧只收集样本，不产生任何判定）
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

        // 框高与深度都要能取到有效中位数，否则本帧仍无判定（下一帧重新攒基线）
        if (!medianPositive(depths, state.baseline_d) ||
            !medianPositive(heights, state.baseline_h)) {
            applyScores(state, out);
            return out;
        }
        state.has_baseline = true;
    }

    // 阶段 2：维护最近 recent_w 帧窗口
    state.recent.push_back(Sample{ height, depth });
    while (static_cast<int>(state.recent.size()) > params_.recent_w) {
        state.recent.pop_front();
    }
    if (static_cast<int>(state.recent.size()) < params_.recent_w) {
        applyScores(state, out);
        return out;
    }

    // 前后半窗中位数：前半窗 = 较早的 recent_w/2 帧，后半窗 = 最近的若干帧
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
        // 该窗口无有效样本（例如卡尔曼滤波后的深度被过冲到 <= 0）：保持上一帧分数
        applyScores(state, out);
        return out;
    }

    // 近期趋势门控：是否“当前仍在靠近”（深度继续降，框高继续涨）
    const bool trend_ok = d_cur_median < d_prev_median && h_cur_median > h_prev_median;

    // 累计变化率（相对基线）与融合分数
    const double depth_drop  = (state.baseline_d - d_cur_median) / state.baseline_d;
    const double height_gain = h_cur_median / state.baseline_h - 1.0;
    const double depth_score = clip01(depth_drop / std::max(params_.thr_depth, kMinThreshold));
    const double scale_score = clip01(height_gain / std::max(params_.thr_height, kMinThreshold));
    const double score       = 0.4 * depth_score + 0.6 * scale_score;

    state.score       = score;
    state.depth_score = depth_score;
    state.scale_score = scale_score;

    // 阶段 3：双边迟滞 + 去抖
    //   进入证据：趋势在靠近 且 score >= score_thr
    //   退出证据：趋势不再靠近 或 score <= exit_score_thr
    //   中间态：保持当前状态（抗闪烁，只有连续证据才切换）
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
    // 三项分数取“最近一次有效计算值”：
    // 本帧无有效窗口时保持上一帧分数，而不是把分数归零，避免下游画面上分数闪烁
    out.score       = static_cast<float>(state.score);
    out.depth_score = static_cast<float>(state.depth_score);
    out.scale_score = static_cast<float>(state.scale_score);
}

// [接近单元装配] 报警白名单 + 两轮车需载人（人 + 车合并为外接大框）

double iouTlwh(const float a[4], const float b[4]) {
    const double ax1 = a[0];
    const double ay1 = a[1];
    const double ax2 = a[0] + a[2];
    const double ay2 = a[1] + a[3];
    const double bx1 = b[0];
    const double by1 = b[1];
    const double bx2 = b[0] + b[2];
    const double by2 = b[1] + b[3];

    const double x1    = std::max(ax1, bx1);
    const double y1    = std::max(ay1, by1);
    const double x2    = std::min(ax2, bx2);
    const double y2    = std::min(ay2, by2);
    const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    if (inter <= 0.0) {
        return 0.0;
    }
    const double uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

bool boxCenterIn(const float inner[4], const float outer[4], double margin) {
    const double cx = inner[0] + 0.5 * inner[2];
    const double cy = inner[1] + 0.5 * inner[3];
    const double dx = margin * outer[2];
    const double dy = margin * outer[3];

    const double ox1 = outer[0];
    const double oy1 = outer[1];
    const double ox2 = outer[0] + outer[2];
    const double oy2 = outer[1] + outer[3];
    return ox1 - dx <= cx && cx <= ox2 + dx && oy1 - dy <= cy && cy <= oy2 + dy;
}

bool personRides(const float person[4], const float two_wheeler[4], double iou_thr) {
    const double px1 = person[0];
    const double py1 = person[1];
    const double px2 = person[0] + person[2];
    const double py2 = person[1] + person[3];
    const double tx1 = two_wheeler[0];
    const double ty1 = two_wheeler[1];
    const double tx2 = two_wheeler[0] + two_wheeler[2];
    const double ty2 = two_wheeler[1] + two_wheeler[3];

    const double ix1   = std::max(px1, tx1);
    const double iy1   = std::max(py1, ty1);
    const double ix2   = std::min(px2, tx2);
    const double iy2   = std::min(py2, ty2);
    const double inter = std::max(0.0, ix2 - ix1) * std::max(0.0, iy2 - iy1);
    if (inter <= 0.0) {
        return false;
    }
    if (iouTlwh(person, two_wheeler) >= iou_thr) {
        return true;
    }
    if (boxCenterIn(person, two_wheeler)) {
        return true;
    }
    // 人框竖长、车框横矮，IoU 很小：用“覆盖率（交叠 / 较小框）”兜底
    const double person_area = std::max((px2 - px1) * (py2 - py1), 1e-6);
    const double bike_area   = std::max((tx2 - tx1) * (ty2 - ty1), 1e-6);
    return inter / std::min(person_area, bike_area) >= 0.25;
}

std::vector<ApproachUnit> assembleApproachUnits(const std::vector<ApproachBoxInput> & tracks,
                                                const ApproachUnitConfig &            config) {
    struct Entry {
        Entry(int id, int cls, const float box_in[4]) : track_id(id), class_id(cls) {
            for (int i = 0; i < 4; ++i) {
                box[i] = box_in[i];
            }
        }

        int   track_id;
        int   class_id;
        float box[4];
    };

    auto makeUnit = [](const Entry & entry, bool merged_rider) {
        ApproachUnit unit;
        unit.unit_id      = entry.track_id;
        unit.class_id     = entry.class_id;
        unit.merged_rider = merged_rider;
        unit.x            = entry.box[0];
        unit.y            = entry.box[1];
        unit.w            = entry.box[2];
        unit.h            = entry.box[3];
        return unit;
    };

    std::vector<ApproachUnit> units;
    units.reserve(tracks.size());

    // use_units == false：不做白名单与载人合并，每个有效目标各自成单元（调试/对比用）
    if (!config.use_units) {
        for (const ApproachBoxInput & track : tracks) {
            if (track.w <= 0.0f || track.h <= 0.0f) {
                continue;
            }
            ApproachUnit unit;
            unit.unit_id  = track.track_id;
            unit.class_id = track.class_id;
            unit.x        = track.x;
            unit.y        = track.y;
            unit.w        = track.w;
            unit.h        = track.h;
            units.push_back(unit);
        }
        return units;
    }

    // 报警白名单为空 => 不限制类别（便于未配置类别映射时仍能工作）
    const bool check_whitelist = !config.alarm_class_ids.empty();

    std::vector<Entry> persons;       // 人（用于载人判定，未认领时也可单独报警）
    std::vector<Entry> two_wheelers;  // 两轮车（自行车/摩托车）
    std::vector<Entry> singles;       // 其它白名单类别（轿车/公交/卡车等）

    for (const ApproachBoxInput & track : tracks) {
        if (track.w <= 0.0f || track.h <= 0.0f) {
            continue;
        }
        if (check_whitelist && !containsId(config.alarm_class_ids, track.class_id)) {
            continue;  // 不在白名单的类别不参与报警
        }
        const float box[4] = { track.x, track.y, track.w, track.h };
        if (containsId(config.person_class_ids, track.class_id)) {
            persons.push_back(Entry(track.track_id, track.class_id, box));
        } else if (containsId(config.two_wheeler_class_ids, track.class_id)) {
            two_wheelers.push_back(Entry(track.track_id, track.class_id, box));
        } else {
            singles.push_back(Entry(track.track_id, track.class_id, box));
        }
    }

    // 第一遍：单目标直接成单元
    for (const Entry & entry : singles) {
        units.push_back(makeUnit(entry, false));
    }

    // 第二遍：两轮车找骑车人（取 IoU 最大且未被认领的），载人则合并为外接大框
    std::vector<bool> person_claimed(persons.size(), false);
    for (const Entry & bike : two_wheelers) {
        int    rider_index = -1;
        double rider_iou   = -1.0;
        for (size_t i = 0; i < persons.size(); ++i) {
            if (person_claimed[i] || !personRides(persons[i].box, bike.box, config.rider_iou)) {
                continue;
            }
            const double iou = iouTlwh(bike.box, persons[i].box);
            if (rider_index < 0 || iou > rider_iou) {
                rider_index = static_cast<int>(i);
                rider_iou   = iou;
            }
        }

        // 有人骑且允许合并：人 + 车合并为外接大框，单元 id 取两轮车的 track id
        // （人框随后仍会单独成单元，见下方第三遍）
        if (rider_index >= 0 && config.merge_person) {
            person_claimed[static_cast<size_t>(rider_index)] = true;
            const Entry & rider = persons[static_cast<size_t>(rider_index)];

            Entry merged  = bike;
            merged.box[0] = std::min(bike.box[0], rider.box[0]);
            merged.box[1] = std::min(bike.box[1], rider.box[1]);
            merged.box[2] =
                std::max(bike.box[0] + bike.box[2], rider.box[0] + rider.box[2]) - merged.box[0];
            merged.box[3] =
                std::max(bike.box[1] + bike.box[3], rider.box[1] + rider.box[3]) - merged.box[1];
            units.push_back(makeUnit(merged, true));
            continue;
        }

        // 两轮车无论有没有骑手都进接近单元（require_rider 规则已停用）
        units.push_back(makeUnit(bike, false));
    }

    // 第三遍：行人各自单独成单元（即使已被两轮车认领合并，也保留自己的判定）
    for (size_t i = 0; i < persons.size(); ++i) {
        units.push_back(makeUnit(persons[i], false));
    }

    return units;
}

}  // namespace approach

// [MotionStateEngine] 对外接口

void MotionStateEngine::configureApproach(const approach::ApproachParams &     params,
                                          approach::FilterMode                 filter_mode,
                                          bool                                 enabled,
                                          const approach::ApproachUnitConfig & unit_config) {
    approach_enabled_     = enabled;
    approach_unit_config_ = unit_config;
    approach_detector_    = approach::ApproachDetectorCumulative(params);
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
        "recent_w={}, score_thr={:.3f}, confirm={}, exit_score_thr={:.3f}, exit_confirm={}, "
        "use_units={}, require_rider={}, rider_iou={:.3f}, merge_person={}",
        approach::filterModeName(filter_mode), params.warmup, params.thr_depth, params.thr_height,
        params.recent_w, params.score_thr, params.confirm, params.exit_score_thr,
        params.exit_confirm, unit_config.use_units, unit_config.require_rider,
        unit_config.rider_iou, unit_config.merge_person);
}

std::vector<approach::ApproachUnit> MotionStateEngine::assembleApproachUnits(
    const std::vector<STrack> & tracks) const {
    if (!approach_enabled_) {
        return std::vector<approach::ApproachUnit>();
    }

    std::vector<approach::ApproachBoxInput> boxes;
    boxes.reserve(tracks.size());
    for (const STrack & track : tracks) {
        if (track.tlwh_.size() < 4) {
            continue;
        }
        approach::ApproachBoxInput box;
        box.track_id = track.track_id_;
        box.class_id = track.class_id_;
        box.x        = track.tlwh_[0];
        box.y        = track.tlwh_[1];
        box.w        = track.tlwh_[2];
        box.h        = track.tlwh_[3];
        boxes.push_back(box);
    }
    return approach::assembleApproachUnits(boxes, approach_unit_config_);
}

approach::ApproachState MotionStateEngine::updateApproachState(int    unit_id,
                                                               float  height,
                                                               float  raw_depth,
                                                               double timestamp) {
    approach::ApproachState state;
    if (!approach_enabled_) {
        return state;
    }

    // 前置因果滤波（只依赖历史）：height 用 log 域卡尔曼更贴合指数增长；无效深度不过滤
    double height_filtered = height;
    double depth_filtered  = raw_depth;
    if (approach_filter_bank_) {
        approach_filter_bank_->update(unit_id, height, raw_depth, timestamp, height_filtered,
                                      depth_filtered);
    }

    state = approach_detector_.update(unit_id, height_filtered, depth_filtered, timestamp);

    // 报警边沿打日志（上升沿 / 下降沿各一条）
    if (state.alarm_started) {
        APP_INFO(
            "[Approach] track {} alarm triggered: score={:.3f} depth_score={:.3f} "
            "scale_score={:.3f}",
            unit_id, state.score, state.depth_score, state.scale_score);
    } else if (state.alarm_cleared) {
        APP_INFO("[Approach] track {} alarm cleared: score={:.3f}", unit_id, state.score);
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

    // 存储当前目标收集到的有效深度点
    std::vector<float> sampled_depths;

    // 在目标框内均匀网格采样，统计有效深度值
    // 采样策略：缩进 20% 边界以避开边缘背景，按 grid_size × grid_size 在框内均匀采点
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

    // 截断均值法：先排序，剔除两端 25% 异常值（前 25% 可能是前景遮挡，后 25% 可能是背景噪声）
    // 再对中间 50% 的数据取均值，得到该目标在当前帧的鲁棒深度估计
    std::sort(sampled_depths.begin(), sampled_depths.end());
    int num_valid = sampled_depths.size();
    if (num_valid < 4) {
        // 数据太少，直接取中位数
        return sampled_depths[num_valid / 2];
    }

    int skip_low  = static_cast<int>(num_valid * 0.25f);  // 剔除25%最近距离（前景毛刺与遮挡）
    int skip_high = static_cast<int>(num_valid * 0.25f);  // 剔除25%最远距离（背景噪声）

    float sum   = 0.0f;
    int   count = 0;
    for (int i = skip_low; i < num_valid - skip_high; ++i) {
        sum += sampled_depths[i];
        count++;
    }

    return count > 0 ? (sum / count) : 0.0f;
}
