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
                               float *      d_cur_src_min_value,
                               float *      d_cur_src_max_value,
                               int          input_w,
                               int          input_h,
                               int          resized_w,
                               int          resized_h,
                               cudaStream_t stream);

void initColorMapTable();  // init color map table

#endif                     // POSTPROCESS_H
