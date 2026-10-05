#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cuda_runtime.h>
#include <opencv2/core/hal/interface.h>

#include <opencv2/opencv.hpp>

// 融合算子：直接从 YOLOv8 channel-first 输出 [numClasses+4, numBboxes] 解码, 过滤低置信度后写为 [1, 1 + maxObjects * numBoxElement]：
// 1: number of valid bboxes; 7: left, top, right, bottom, confidence, class, keepflag
void decode(const float * src,
            float *       dst,
            int           numBboxes,
            int           numClasses,
            float         confThresh,
            int           maxObjects,
            int           numBoxElement,
            cudaStream_t  stream);

void nms(float * data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream);

void normalize_colormap_resize(float *      src,
                               uchar *      norm_depth,
                               uchar3 *     norm_colormap,
                               uchar *      dst_depth,
                               uchar3 *     dst_colormap,
                               int          input_w,
                               int          input_h,
                               int          resized_w,
                               int          resized_h,
                               cudaStream_t stream);

void initColorMapTable();  // INFERNO 颜色映射表初始化，仅需调用一次

// 生成 256 色 TURBO 表并上传到 __constant__ 内存
void initTurboColorTable();

// float 深度转 TURBO 伪彩, 去 letterbox 内容区 ROI → 双线性 resize 到原始分辨率 → 对有限值求 P1/P99 分位数 → clip 归一化 → 查 TURBO 表。
#define DEPTH_COLORMAP_STAT_BLOCKS 64
void floatDepthColormapResize(const float * src,
                              int           in_w,
                              int           roi_x,
                              int           roi_y,
                              int           roi_w,
                              int           roi_h,
                              float *       stage_float,  // [out_w*out_h] float
                              int           out_w,
                              int           out_h,
                              uchar *       dst_gray,        // [out_w*out_h] uchar
                              uchar3 *      dst_color,       // [out_w*out_h] uchar3 (BGR)
                              float *       stat_min,        // [DEPTH_COLORMAP_STAT_BLOCKS]
                              float *       stat_max,        // [DEPTH_COLORMAP_STAT_BLOCKS]
                              int *         stat_count,      // [DEPTH_COLORMAP_STAT_BLOCKS]
                              float *       range_2f,        // [2]: {min, max}
                              int *         hist,            // [256]
                              float *       percentiles_2f,  // [2]: {P1, P99}
                              int           stat_blocks,
                              cudaStream_t  stream);

__inline__ void scale_bbox(const cv::Mat & img, float bbox[4], int input_w, int input_h) {
    float r_w   = input_w / (img.cols * 1.0);
    float r_h   = input_h / (img.rows * 1.0);
    float r     = std::min(r_w, r_h);
    float pad_h = (input_h - r * img.rows) / 2;
    float pad_w = (input_w - r * img.cols) / 2;

    bbox[0] = (bbox[0] - pad_w) / r;
    bbox[1] = (bbox[1] - pad_h) / r;
    bbox[2] = (bbox[2] - pad_w) / r;
    bbox[3] = (bbox[3] - pad_h) / r;
}

#endif  // POSTPROCESS_H
