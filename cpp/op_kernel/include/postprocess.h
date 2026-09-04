#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cuda_runtime.h>
#include <opencv2/core/hal/interface.h>

#include <opencv2/opencv.hpp>

void transpose(float * src, float * dst, int numBboxes, int numElements, cudaStream_t stream);
/*
    transpose [1 84 8400] convert to [1 8400 84]
src:          Tensor, dim is [1 84 8400]
dst:          Tensor, dim is [1 8400 84]
numBboxes:    number of bboxes
numElements:  center_x, center_y, width, height, 80 or other classes
*/

void decode(float *      src,
            float *      dst,
            int          numBboxes,
            int          numClasses,
            float        confThresh,
            int          maxObjects,
            int          numBoxElement,
            cudaStream_t stream);
/*
    convert [1 8400 84] to [1 7001](7001 = 1 + 1000 * 7, 1: number of valid
   bboxes 1000: max bboxes, valid bboxes may less than 1000, 7: left, top,
   right, bottom, confidence, class, keepflag)
*/

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

// 用 cv::applyColorMap(COLORMAP_TURBO) 生成 256 色 TURBO 表并上传到 __constant__ 内存
void initTurboColorTable();

// float 深度转 TURBO 伪彩（YoloDepthModel 使用，全异步、无主机同步）。
// 语义对齐 Python depth_to_colormap：去 letterbox 内容区 ROI → 双线性 resize 到原始分辨率
// → 对有限值求 P1/P99 分位数 → clip 归一化 → 查 TURBO 表。
// stage_float / stat_min / stat_max / stat_count / range_2f / hist / percentiles_2f
// 均为调用方持有的设备端中间缓冲（每帧复用）。
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
