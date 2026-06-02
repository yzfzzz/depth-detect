#include "preprocess.h"

__global__ void letterbox(const uchar * srcData,
                          const int     srcH,
                          const int     srcW,
                          uchar *       tgtData,
                          const int     tgtH,
                          const int     tgtW,
                          const int     rszH,
                          const int     rszW,
                          const int     startY,
                          const int     startX) {
    int ix   = threadIdx.x + blockDim.x * blockIdx.x;
    int iy   = threadIdx.y + blockDim.y * blockIdx.y;
    int idx  = ix + iy * tgtW;
    int idx3 = idx * 3;

    if (ix > tgtW || iy > tgtH) {
        return;  // thread out of target range
    }
    // gray region on target image
    if (iy < startY || iy > (startY + rszH - 1)) {
        tgtData[idx3]     = 128;
        tgtData[idx3 + 1] = 128;
        tgtData[idx3 + 2] = 128;
        return;
    }
    if (ix < startX || ix > (startX + rszW - 1)) {
        tgtData[idx3]     = 128;
        tgtData[idx3 + 1] = 128;
        tgtData[idx3 + 2] = 128;
        return;
    }

    float scaleY = (float) rszH / (float) srcH;
    float scaleX = (float) rszW / (float) srcW;

    // (ix,iy)为目标图像坐标
    // (before_x,before_y)原图坐标
    float beforeX = float(ix - startX + 0.5) / scaleX - 0.5;
    float beforeY = float(iy - startY + 0.5) / scaleY - 0.5;
    // 原图像坐标四个相邻点
    // 获得变换前最近的四个顶点,取整
    int   topY    = static_cast<int>(beforeY);
    int   bottomY = topY + 1;
    int   leftX   = static_cast<int>(beforeX);
    int   rightX  = leftX + 1;
    //计算变换前坐标的小数部分
    float u       = beforeX - leftX;
    float v       = beforeY - topY;

    if (topY >= srcH - 1 && leftX >= srcW - 1)  //右下角
    {
        for (int k = 0; k < 3; k++) {
            tgtData[idx3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k];
        }
    } else if (topY >= srcH - 1)  // 最后一行
    {
        for (int k = 0; k < 3; k++) {
            tgtData[idx3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k] +
                                (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k];
        }
    } else if (leftX >= srcW - 1)  // 最后一列
    {
        for (int k = 0; k < 3; k++) {
            tgtData[idx3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k] +
                                (1. - u) * (v) *srcData[(leftX + bottomY * srcW) * 3 + k];
        }
    } else  // 非最后一行或最后一列情况
    {
        for (int k = 0; k < 3; k++) {
            tgtData[idx3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k] +
                                (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k] +
                                (1. - u) * (v) *srcData[(leftX + bottomY * srcW) * 3 + k] +
                                u * v * srcData[(rightX + bottomY * srcW) * 3 + k];
        }
    }
}

__global__ void process(const uchar * srcData, float * tgtData, const int h, const int w) {
    int ix   = threadIdx.x + blockIdx.x * blockDim.x;
    int iy   = threadIdx.y + blockIdx.y * blockDim.y;
    int idx  = ix + iy * w;
    int idx3 = idx * 3;

    if (ix < w && iy < h) {
        tgtData[idx]             = (float) srcData[idx3 + 2] / 255.0;  // R pixel
        tgtData[idx + h * w]     = (float) srcData[idx3 + 1] / 255.0;  // G pixel
        tgtData[idx + h * w * 2] = (float) srcData[idx3] / 255.0;      // B pixel
    }
}

void preprocess(const cv::Mat & srcImg,
                float *         dstDevData,
                uchar *         srcDevData,
                uchar *         midDevData,
                int             raw_img_h,
                int             raw_img_w,
                int             input_h,
                int             input_w,
                cudaStream_t    stream) {
    // calculate width and height after resize
    int   w, h, x, y;
    float r_w = input_w / (raw_img_w * 1.0);
    float r_h = input_h / (raw_img_h * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * raw_img_h;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * raw_img_w;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }

    cudaMemcpyAsync(srcDevData, srcImg.data, sizeof(uchar) * raw_img_h * raw_img_w * 3,
                    cudaMemcpyHostToDevice, stream);

    dim3 blockSize(32, 32);
    dim3 gridSize((input_w + blockSize.x - 1) / blockSize.x,
                  (input_h + blockSize.y - 1) / blockSize.y);

    // letterbox and resize
    letterbox<<<gridSize, blockSize, 0, stream>>>(srcDevData, raw_img_h, raw_img_w, midDevData,
                                                  input_h, input_w, h, w, y, x);
    // hwc to chw / bgr to rgb / normalize
    process<<<gridSize, blockSize, 0, stream>>>(midDevData, dstDevData, input_h, input_w);
}

void preprocess_v2(float *      dstDevData,
                   uchar *      srcDevData,
                   uchar *      midDevData,
                   int          raw_img_h,
                   int          raw_img_w,
                   int          input_h,
                   int          input_w,
                   cudaStream_t stream) {
    // calculate width and height after resize
    int   w, h, x, y;
    float r_w = input_w / (raw_img_w * 1.0);
    float r_h = input_h / (raw_img_h * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * raw_img_h;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * raw_img_w;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }

    dim3 blockSize(32, 32);
    dim3 gridSize((input_w + blockSize.x - 1) / blockSize.x,
                  (input_h + blockSize.y - 1) / blockSize.y);

    // letterbox and resize
    letterbox<<<gridSize, blockSize, 0, stream>>>(srcDevData, raw_img_h, raw_img_w, midDevData,
                                                  input_h, input_w, h, w, y, x);
    // hwc to chw / bgr to rgb / normalize
    process<<<gridSize, blockSize, 0, stream>>>(midDevData, dstDevData, input_h, input_w);
}

__global__ void resize_mat2tensor_norm_kernel(uchar * src,
                                              float * dst,
                                              int     input_w,
                                              int     input_h,
                                              int     resized_w,
                                              int     resized_h,
                                              float   resize_scale_w,
                                              float   resize_scale_h,
                                              float * mean,
                                              float * std) {
    int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_idx >= resized_w || dst_idy >= resized_h) {
        return;
    }

    float resize_src_x;
    float resize_src_y;
    int   src_idx;
    int   src_idy;

    // scale is src/dst, i.e. scale > 1, image will be smaller than before
    // CentralAligned
    resize_src_x = (dst_idx + 0.5f) * resize_scale_w - 0.5f;
    resize_src_y = (dst_idy + 0.5f) * resize_scale_h - 0.5f;

    src_idx = (int) floorf(resize_src_x);
    src_idy = (int) floorf(resize_src_y);

    resize_src_x = resize_src_x - src_idx;
    resize_src_y = resize_src_y - src_idy;
    float fx1y1  = resize_src_x * resize_src_y;
    float fx0y0  = 1.0f - resize_src_x - resize_src_y + fx1y1;
    float fx1y0  = resize_src_x - fx1y1;
    float fx0y1  = resize_src_y - fx1y1;

#pragma unroll
    // resize + bgr2rgb + norm + chw
    for (int c = 0; c < 3; ++c) {
        // clamp indices used for neighbours
        int sx0 = min(max(src_idx, 0), input_w - 1);
        int sy0 = min(max(src_idy, 0), input_h - 1);
        int sx1 = min(sx0 + 1, input_w - 1);
        int sy1 = min(sy0 + 1, input_h - 1);

        // read BGR from src but map to dst channel c as RGB:
        // read channel (2 - c) from src (so c==0 gets R)
        float p00 = src[(sy0 * input_w + sx0) * 3 + (2 - c)];
        float p10 = src[(sy0 * input_w + sx1) * 3 + (2 - c)];
        float p01 = src[(sy1 * input_w + sx0) * 3 + (2 - c)];
        float p11 = src[(sy1 * input_w + sx1) * 3 + (2 - c)];

        float val = p00 * fx0y0 + p10 * fx1y0 + p01 * fx0y1 + p11 * fx1y1;

        // normalize and write to CHW float dst
        int out_idx  = c * resized_h * resized_w + dst_idy * resized_w + dst_idx;
        dst[out_idx] = (val / 255.0f - mean[c]) / std[c];
    }
}

void depthPreprocess(uchar *      src,
                     float *      dst,
                     int          input_w,
                     int          input_h,
                     int          resized_w,
                     int          resized_h,
                     float *      mean,
                     float *      std,
                     cudaStream_t stream) {
    dim3 blockSize(32, 8);
    dim3 gridSize((resized_w + 31) >> 5, (resized_h + 7) >> 3);

    // mat2tensor = bgr2rgb、hwc2chw
    resize_mat2tensor_norm_kernel<<<gridSize, blockSize, 0, stream>>>(
        src, dst, input_w, input_h, resized_w, resized_h, (float) input_w / resized_w,
        (float) input_h / resized_h, mean, std);
}
