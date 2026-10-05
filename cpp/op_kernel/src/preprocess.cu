#include "preprocess.h"

// 融合算子：letterbox + 双线性 resize + BGR→RGB + /255 归一化 + HWC→CHW
// 直接从原始 BGR 图写出 float CHW 张量，省去中间 uchar 缓冲的一次显存往返
__global__ void letterbox_norm_kernel(const uchar * src_data,
                                      const int     srcH,
                                      const int     srcW,
                                      float *       tgt_data,
                                      const int     tgtH,
                                      const int     tgtW,
                                      const int     rszH,
                                      const int     rszW,
                                      const int     startY,
                                      const int     startX) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    if (ix >= tgtW || iy >= tgtH) {
        return;
    }

    int idx   = iy * tgtW + ix;
    int plane = tgtH * tgtW;

    // 灰边填充区域：实图像范围外的像素填充灰色 (128,128,128)，归一化后为 128/255
    if (iy < startY || iy > (startY + rszH - 1) || ix < startX || ix > (startX + rszW - 1)) {
        float gray                = 128.0f / 255.0f;
        tgt_data[idx]             = gray;  // R
        tgt_data[idx + plane]     = gray;  // G
        tgt_data[idx + plane * 2] = gray;  // B
        return;
    }

    float scaleY = (float) rszH / (float) srcH;
    float scaleX = (float) rszW / (float) srcW;

    // 中心对齐反向映射：目标坐标 → 源坐标，+0.5 偏移避免边缘伪影
    float beforeX = float(ix - startX + 0.5) / scaleX - 0.5;
    float beforeY = float(iy - startY + 0.5) / scaleY - 0.5;
    // 双线性插值：计算源坐标左上整像素及小数偏移
    int   srcX    = (int) floorf(beforeX);
    int   srcY    = (int) floorf(beforeY);
    float u       = beforeX - srcX;
    float v       = beforeY - srcY;
    float fx1y1   = u * v;
    float fx0y0   = 1.0f - u - v + fx1y1;
    float fx1y0   = u - fx1y1;
    float fx0y1   = v - fx1y1;

    // 合并 resize + BGR→RGB + 归一化 + HWC→CHW，遍历三个通道
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        // 边界 clamp 防止越界
        int sx0 = min(max(srcX, 0), srcW - 1);
        int sy0 = min(max(srcY, 0), srcH - 1);
        int sx1 = min(sx0 + 1, srcW - 1);
        int sy1 = min(sy0 + 1, srcH - 1);

        // BGR→RGB 通道重映射：c=0→R(读src channel 2), c=1→G(读src channel 1), c=2→B(读src channel 0)
        float p00 = src_data[(sy0 * srcW + sx0) * 3 + (2 - c)];
        float p10 = src_data[(sy0 * srcW + sx1) * 3 + (2 - c)];
        float p01 = src_data[(sy1 * srcW + sx0) * 3 + (2 - c)];
        float p11 = src_data[(sy1 * srcW + sx1) * 3 + (2 - c)];

        float val = p00 * fx0y0 + p10 * fx1y0 + p01 * fx0y1 + p11 * fx1y1;

        // 归一化后写入 CHW 布局：out[c * H * W + y * W + x]
        tgt_data[c * plane + idx] = val / 255.0f;
    }
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

void yoloPreprocess(float *      dst_dev_data,
                    uchar *      src_dev_data,
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

    dim3 blockSize(32, 8);
    dim3 gridSize((input_w + blockSize.x - 1) / blockSize.x,
                  (input_h + blockSize.y - 1) / blockSize.y);

    letterbox_norm_kernel<<<gridSize, blockSize, 0, stream>>>(
        src_dev_data, raw_img_h, raw_img_w, dst_dev_data, input_h, input_w, h, w, y, x);
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
