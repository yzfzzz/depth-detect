#pragma once

#include "base_model.h"
#include "memory.h"
#include "public.h"

#include <opencv2/core/hal/interface.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

struct DepthInferOutputSet {
    uchar *  depth_output_data;
    uchar3 * depth_colormap_data;
};

// 深度估计模型，继承自 BaseModel，支持 TensorRT 和 ONNX Runtime 后端
class DepthModel : public BaseModel {
  public:
    DepthModel();
    ~DepthModel() override;

    // 初始化模型
    bool init(const std::string & model_path, int raw_img_w, int raw_img_h, bool is_normalize);

    // 同步推理
    bool runInference(FrameInputContext & frame_input_context) override;

    // 异步推理
    bool runInferenceAsync(FrameInputContext & frame_input_context) override;
    void getInferOutputResult(InferOutputContext & infer_output_context) override;

  private:
    // BaseModel 接口实现
    // opencv 预处理和后处理（用于 ONNX Runtime）
    void cvMatPreProcess(FrameInputContext & frame_input_context) override;
    void cvMatPostProcess(FrameInputContext & frame_input_context) override;
    // CUDA 预处理和后处理（用于 TensorRT）
    void cudaPreProcess(FrameInputContext & frame_input_context) override;
    void cudaPostProcess(FrameInputContext & frame_input_context) override;


  private:
    bool is_normalize_;

    // 归一化参数
    std::vector<float> h_mean_;
    std::vector<float> h_std_;

    // CUDA 资源（仅 TensorRT 后端使用）
    std::array<unique_ptr_cuda<void>, 2> d_buffer_;
    unique_ptr_cuda<uchar>               d_buffer_norm_depth_;
    unique_ptr_cuda<uchar3>              d_buffer_norm_colormap_;
    unique_ptr_cuda<float>               d_depth_infer_min_value_;
    unique_ptr_cuda<float>               d_depth_infer_max_value_;
    unique_ptr_cuda<void>                d_cub_mid_min_;
    unique_ptr_cuda<void>                d_cub_mid_max_;
    size_t                               cub_max_bytes_ = 0;
    size_t                               cub_min_bytes_ = 0;
    size_t                               cub_bytes_     = 0;
    std::vector<float>                   h_output_data_;
    unique_ptr_cuda<float>               d_mean_;
    unique_ptr_cuda<float>               d_std_;
    unique_ptr_cuda<uchar>               d_before_preprocess_img_data_;

    unique_ptr_pinned_cuda<uchar>  host_pinned_depth_output_data_;
    unique_ptr_pinned_cuda<uchar3> host_pinned_depth_colormap_data_;
    unique_ptr_cuda<uchar>         d_buffer_dst_depth_;
    unique_ptr_cuda<uchar3>        d_buffer_dst_colormap_;

    // 颜色映射表
    cv::Mat colormap_table_;
};
