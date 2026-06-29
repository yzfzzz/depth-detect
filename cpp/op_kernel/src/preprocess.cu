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
        return;
    }
    // 灰边填充区域：实图像范围外的像素填充灰色 (128,128,128)
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

    // 中心对齐反向映射：目标坐标 → 源坐标，+0.5 偏移避免边缘伪影
    float beforeX = float(ix - startX + 0.5) / scaleX - 0.5;
    float beforeY = float(iy - startY + 0.5) / scaleY - 0.5;
    // 双线性插值：计算源坐标相邻四个整像素及小数偏移
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
        tgtData[idx]             = (float) srcData[idx3 + 2] / 255.0;  // BGR→RGB: R=src[2]
        tgtData[idx + h * w]     = (float) srcData[idx3 + 1] / 255.0;  // G=src[1]
        tgtData[idx + h * w * 2] = (float) srcData[idx3] / 255.0;      // B=src[0]
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
    // Letterbox 计算：选择较小缩放比例保持宽高比，不足部分居中填充灰边
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

    // GPU 预处理流水线：letterbox→HWC2CHW/BGR2RGB/归一化，同一 stream 顺序执行
    letterbox<<<gridSize, blockSize, 0, stream>>>(srcDevData, raw_img_h, raw_img_w, midDevData,
                                                  input_h, input_w, h, w, y, x);
    process<<<gridSize, blockSize, 0, stream>>>(midDevData, dstDevData, input_h, input_w);
}

// preprocess_v2：与 preprocess 逻辑相同，但输入已是 GPU 端数据，省略 H2D 拷贝
void preprocess_v2(float *      dstDevData,
                   uchar *      srcDevData,
                   uchar *      midDevData,
                   int          raw_img_h,
                   int          raw_img_w,
                   int          input_h,
                   int          input_w,
                   cudaStream_t stream) {
    // Letterbox 计算：选择较小缩放比例保持宽高比，不足部分居中填充灰边
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

    letterbox<<<gridSize, blockSize, 0, stream>>>(srcDevData, raw_img_h, raw_img_w, midDevData,
                                                  input_h, input_w, h, w, y, x);
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

    // 中心对齐反向映射：目标坐标 → 源坐标，+0.5 偏移避免边缘像素偏差
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

    // 合并 resize + BGR→RGB + 归一化 + HWC→CHW，遍历三个通道
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        // 边界 clamp 防止越界
        int sx0 = min(max(src_idx, 0), input_w - 1);
        int sy0 = min(max(src_idy, 0), input_h - 1);
        int sx1 = min(sx0 + 1, input_w - 1);
        int sy1 = min(sy0 + 1, input_h - 1);

        // BGR→RGB 通道重映射：c=0→R(读src channel 2), c=1→G(读src channel 1), c=2→B(读src channel 0)
        float p00 = src[(sy0 * input_w + sx0) * 3 + (2 - c)];
        float p10 = src[(sy0 * input_w + sx1) * 3 + (2 - c)];
        float p01 = src[(sy1 * input_w + sx0) * 3 + (2 - c)];
        float p11 = src[(sy1 * input_w + sx1) * 3 + (2 - c)];

        float val = p00 * fx0y0 + p10 * fx1y0 + p01 * fx0y1 + p11 * fx1y1;

        // 归一化后写入 CHW 布局：out[c * H * W + y * W + x]
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

    // 深度图预处理：resize + BGR→RGB + 归一化，一步完成，与 YOLO 预处理共享同一 kernel
    resize_mat2tensor_norm_kernel<<<gridSize, blockSize, 0, stream>>>(
        src, dst, input_w, input_h, resized_w, resized_h, (float) input_w / resized_w,
        (float) input_h / resized_h, mean, std);
}
