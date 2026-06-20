#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"

#include <opencv2/opencv.hpp>

// main ./video_path ./yolo_fp16.trt ./yolo_fp32.trt ./depth_fp16.trt ./depth_fp32.trt

// ============================================================================
// 误差度量工具函数
// ============================================================================

// 计算两个 cv::Mat 的深度误差指标
struct DepthErrorMetrics {
    double mae      = 0.0;  // Mean Absolute Error
    double rmse     = 0.0;  // Root Mean Square Error
    double max_abs  = 0.0;  // 最大绝对误差
    double rel_err  = 0.0;  // 平均相对误差: |fp16 - fp32| / (|fp32| + eps)
    int    valid_px = 0;    // 有效像素数
};

DepthErrorMetrics computeDepthError(const std::vector<float> & depth_fp16,
                                    const std::vector<float> & depth_fp32) {
    DepthErrorMetrics metrics;
    if (depth_fp16.empty() || depth_fp32.empty() || depth_fp16.size() != depth_fp32.size()) {
        APP_ERROR("Depth vector size mismatch: fp16({}) vs fp32({})", depth_fp16.size(),
                  depth_fp32.size());
        return metrics;
    }

    double       sum_abs = 0.0, sum_sq = 0.0, sum_rel = 0.0;
    double       max_val = 0.0;
    const double eps     = 1e-6;
    for (int i = 0; i < depth_fp16.size() && i < depth_fp32.size(); ++i) {
        float v16 = depth_fp16[i];
        float v32 = depth_fp32[i];
        if (std::isfinite(v16) && std::isfinite(v32) && v16 > 0 && v32 > 0) {
            double diff = std::abs(v16 - v32);
            sum_abs += diff;
            sum_sq += diff * diff;
            sum_rel += diff / (std::abs(v32) + eps);
            max_val = std::max(max_val, diff);
            metrics.valid_px++;
        }
    }

    if (metrics.valid_px > 0) {
        metrics.mae     = sum_abs / metrics.valid_px;
        metrics.rmse    = std::sqrt(sum_sq / metrics.valid_px);
        metrics.max_abs = max_val;
        metrics.rel_err = sum_rel / metrics.valid_px;
    }
    return metrics;
}

// 计算 BBox 的 IoU
float computeIoU(const std::array<float, 4> & a, const std::array<float, 4> & b) {
    float x1      = std::max(a[0], b[0]);
    float y1      = std::max(a[1], b[1]);
    float x2      = std::min(a[2], b[2]);
    float y2      = std::min(a[3], b[3]);
    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter   = inter_w * inter_h;
    float area_a  = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b  = (b[2] - b[0]) * (b[3] - b[1]);
    float iou     = inter / (area_a + area_b - inter + 1e-6f);
    return iou;
}

// 对比两帧 YOLO 检测结果的误差
struct YoloDetectionError {
    double avg_iou        = 0.0;  // 配对检测框平均 IoU
    double avg_conf_diff  = 0.0;  // 平均置信度差
    int    class_mismatch = 0;    // 类别不匹配次数
    int    count_diff     = 0;    // 检测数量差异
    int    paired_count   = 0;    // 成功配对的检测框数
};

YoloDetectionError computeDetectionError(const std::vector<Detection> & dets_fp16,
                                         const std::vector<Detection> & dets_fp32) {
    YoloDetectionError err;
    err.count_diff = std::abs((int) dets_fp16.size() - (int) dets_fp32.size());

    // 按 class + 近似 bbox 贪心配对
    std::vector<bool> matched_fp32(dets_fp32.size(), false);
    double            sum_iou = 0.0, sum_conf = 0.0;

    for (const auto & d16 : dets_fp16) {
        float best_iou = 0.5f;  // IoU 阈值
        int   best_idx = -1;
        for (size_t j = 0; j < dets_fp32.size(); ++j) {
            if (matched_fp32[j]) {
                continue;
            }
            if (d16.classId != dets_fp32[j].classId) {
                continue;
            }
            float iou = computeIoU(d16.bbox, dets_fp32[j].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = static_cast<int>(j);
            }
        }
        if (best_idx >= 0) {
            matched_fp32[best_idx] = true;
            sum_iou += best_iou;
            sum_conf += std::abs(d16.conf - dets_fp32[best_idx].conf);
            err.paired_count++;
        }
    }

    if (err.paired_count > 0) {
        err.avg_iou       = sum_iou / err.paired_count;
        err.avg_conf_diff = sum_conf / err.paired_count;
    }

    // 统计类别不匹配
    std::unordered_map<int, int> cls_cnt_fp16, cls_cnt_fp32;
    for (auto & d : dets_fp16) {
        cls_cnt_fp16[d.classId]++;
    }
    for (auto & d : dets_fp32) {
        cls_cnt_fp32[d.classId]++;
    }
    for (auto element : cls_cnt_fp16) {
        int cls   = element.first;
        int cnt16 = element.second;
        err.class_mismatch += std::abs(cnt16 - cls_cnt_fp32[cls]);
    }

    return err;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char ** argv) {
    if (argc != 6) {
        APP_ERROR(
            "Usage: {} <video_path> <yolo_fp16.trt> <yolo_fp32.trt> <depth_fp16.trt> "
            "<depth_fp32.trt>",
            argv[0]);
    }
    std::string video_path      = argv[1];
    std::string yolo_fp16_path  = argv[2];
    std::string yolo_fp32_path  = argv[3];
    std::string depth_fp16_path = argv[4];
    std::string depth_fp32_path = argv[5];

    ConfigManager config_manager("./config.yaml");
    // 初始化日志系统
    LoggerManager::getInstance(config_manager);

    APP_INFO("Application started with video: {}", std::string(video_path));
    IOManager io_manager("video", "none");
    FrameMeta frame_meta = io_manager.Init(video_path);

    Pipeline pipeline_fp16(depth_fp16_path, yolo_fp16_path, frame_meta);
    Pipeline pipeline_fp32(depth_fp32_path, yolo_fp32_path, frame_meta);

    int                num_frames = 0;
    FrameInputContext  frame_input_context(num_frames, frame_meta);
    InferOutputContext infer_output_context_fp16;
    InferOutputContext infer_output_context_fp32;

    // ---- 累积误差统计 ----
    DepthErrorMetrics  depth_err_sum;
    YoloDetectionError yolo_err_sum;
    int                processed_frames = 0;

    while (true) {
        frame_input_context.setFrameID(num_frames);

        if (!io_manager.readNextFrame(frame_input_context, false) ||
            frame_input_context.raw_img.empty()) {
            break;
        }
        num_frames++;

        pipeline_fp16.processOverlap(frame_input_context, infer_output_context_fp16);
        pipeline_fp32.processOverlap(frame_input_context, infer_output_context_fp32);

        // ---- 逐帧计算误差 ----
        // 1. 深度误差
        DepthErrorMetrics depth_frame_err =
            computeDepthError(infer_output_context_fp16.depth_raw_infer_out,
                              infer_output_context_fp32.depth_raw_infer_out);
        depth_err_sum.mae += depth_frame_err.mae;
        depth_err_sum.rmse += depth_frame_err.rmse;
        depth_err_sum.max_abs = std::max(depth_err_sum.max_abs, depth_frame_err.max_abs);
        depth_err_sum.rel_err += depth_frame_err.rel_err;
        depth_err_sum.valid_px += depth_frame_err.valid_px;

        // 2. YOLO 检测误差
        YoloDetectionError yolo_frame_err = computeDetectionError(
            infer_output_context_fp16.detections, infer_output_context_fp32.detections);
        yolo_err_sum.avg_iou += yolo_frame_err.avg_iou;
        yolo_err_sum.avg_conf_diff += yolo_frame_err.avg_conf_diff;
        yolo_err_sum.class_mismatch += yolo_frame_err.class_mismatch;
        yolo_err_sum.count_diff += yolo_frame_err.count_diff;
        yolo_err_sum.paired_count += yolo_frame_err.paired_count;

        processed_frames++;

        if (num_frames > 1000) {
            break;
        }
    }

    // ========================================================================
    // 输出量化误差报告
    // ========================================================================
    APP_INFO("========== FP16 vs FP32 Quantization Error Report ==========");
    APP_INFO("Total frames processed: {}", processed_frames);

    // --- 深度误差 ---
    APP_INFO("--- Depth Map Error ---");
    APP_INFO("  MAE:                {:.6f}", depth_err_sum.mae / processed_frames);
    APP_INFO("  RMSE:               {:.6f}", depth_err_sum.rmse / processed_frames);
    APP_INFO("  Max Absolute Error: {:.6f}", depth_err_sum.max_abs);
    APP_INFO("  Relative Error:     {:.2f}%", depth_err_sum.rel_err / processed_frames * 100.0);

    // --- YOLO 检测误差 ---
    APP_INFO("--- YOLO Detection Error ---");
    if (yolo_err_sum.paired_count > 0) {
        APP_INFO("  Avg IoU (paired):   {:.4f}", yolo_err_sum.avg_iou / processed_frames);
        APP_INFO("  Avg Conf Diff:      {:.6f}", yolo_err_sum.avg_conf_diff / processed_frames);
    }
    APP_INFO("  Class Mismatches:   {} (total)", yolo_err_sum.class_mismatch);
    APP_INFO("  Count Diffs:        {} (total)", yolo_err_sum.count_diff);

    APP_INFO("=============================================================\n");
    return 0;
}
