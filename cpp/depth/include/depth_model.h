#pragma once

#include "memory.h"
#include "public.h"

#include <cuda_runtime_api.h>
#include <logger_manager.h>
#include <NvInfer.h>
#include <opencv2/core/hal/interface.h>

#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

class DepthModel {
  public:
    DepthModel();

    virtual ~DepthModel();

    // 通用的 Engine 加载和显存分配流程
    virtual void init(const std::string & engine_path, int img_w, int img_h, bool is_normalize);

    // 同步推理
    virtual std::pair<cv::Mat, cv::Mat> predict(const cv::Mat & image);
    // 异步推理
    virtual void                        predictAsync(uchar * d_image);
    virtual void                        waitAsync();
    virtual std::pair<cv::Mat, cv::Mat> getPredictResultAsync();

    // 子类必须实现的专属预处理
    virtual std::vector<float> preProcess(const cv::Mat & image);
    virtual void               preProcessAsync(uchar * d_image);
    std::vector<float>         h_mean_;
    std::vector<float>         h_std_;


  protected:
    TrtRuntimePtr                        runtime_;
    TrtEnginePtr                         engine_;
    TrtContextPtr                        context_;
    cudaStream_t                         stream_;
    std::string                          io_tensor_name_[2]{ "input", "output" };
    std::array<unique_ptr_cuda<void>, 2> d_buffer_;

    unique_ptr_cuda<uchar> d_buffer_norm_depth_;  // 经过正则化的深度图，size = img_h*img_w
    unique_ptr_cuda<uchar3> d_buffer_norm_colormap_;  // 颜色图,size = img_h*img_w*3
    unique_ptr_cuda<uchar> d_buffer_dst_depth_;  // 经过正则化的深度图，size = raw_img_w*raw_img_h
    unique_ptr_cuda<uchar3> d_buffer_dst_colormap_;  // 颜色图,size = raw_img_w*raw_img_h*3
    unique_ptr_cuda<float>  d_depth_infer_min_value_;
    unique_ptr_cuda<float>  d_depth_infer_max_value_;
    unique_ptr_cuda<void>   d_cub_mid_min_;
    unique_ptr_cuda<void>   d_cub_mid_max_;
    size_t                  cub_max_bytes_ = 0;
    size_t                  cub_min_bytes_ = 0;
    size_t                  cub_bytes_     = 0;
    std::vector<float>      h_output_data_;
    unique_ptr_cuda<float>  d_mean_;
    unique_ptr_cuda<float>  d_std_;
    unique_ptr_cuda<uchar>  d_before_preprocess_img_data_;

    int                            input_h_, input_w_;      // 模型输出的尺寸
    int                            raw_img_w_, raw_img_h_;  // 原始输入图像的尺寸
    unique_ptr_pinned_cuda<uchar>  host_pinned_depth_output_data_;
    unique_ptr_pinned_cuda<uchar3> host_pinned_depth_colormap_data_;
    Logger                         logger_;
};
