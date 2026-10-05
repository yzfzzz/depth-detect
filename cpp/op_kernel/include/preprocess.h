#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

void yoloPreprocess(float *      dst_dev_data,
                    uchar *      src_dev_data,
                    int          raw_img_h,
                    int          raw_img_w,
                    int          input_h,
                    int          input_w,
                    cudaStream_t stream);

void depthPreprocess(uchar *      src,
                     float *      dst,
                     int          input_w,
                     int          input_h,
                     int          resized_w,
                     int          resized_h,
                     float *      mean,
                     float *      std,
                     cudaStream_t stream);

#endif  // PREPROCESS_H
