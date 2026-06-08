#pragma once

#include "base_model.h"
#include "frame.h"
#include "memory.h"
#include "public.h"

#include <opencv2/core/hal/interface.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

// 深度估计模型，继承自 BaseModel，支持 TensorRT 和 ONNX Runtime 后端
class DepthModel : public BaseModel {
  public:
    DepthModel();
    ~DepthModel() override;

    // 初始化模型
    bool init(std::map<std::string, std::string> model_path,
              int                                raw_img_w,
              int                                raw_img_h,
              bool                               is_normalize,
              bool                               use_gpu = false);

    void getInferOutputResult(InferOutputContext & infer_output_context) override;

  private:
    // BaseModel 接口实现
    std::vector<float> cvMatPreProcess(FrameInputContext & frame_input_context) override;
    void               cvMatPostProcess(InferOutputContext & infer_output_context) override;
    void               cudaPreProcess(FrameInputContext & frame_input_context) override;
    void               cudaPostProcess(FrameInputContext & frame_input_context) override;

  private:
    bool               is_normalize_;
    std::vector<float> h_mean_;
    std::vector<float> h_std_;

    // ── CUDA 资源（仅 TensorRT 后端使用）──

    // 后处理中间 buffer：归一化 depth + colormap（模型分辨率）
    unique_ptr_cuda<uchar>  d_buffer_norm_depth_;
    unique_ptr_cuda<uchar3> d_buffer_norm_colormap_;

    // 后处理归约：min/max 标量 + CUB 临时空间（min/max 串行复用同一 buffer）
    unique_ptr_cuda<float> d_depth_minmax_;  // float[2]: [min, max]
    unique_ptr_cuda<void>  d_cub_temp_;
    size_t                 cub_temp_bytes_ = 0;

    // 预处理参数：mean[3] + std[3] 合并存储
    unique_ptr_cuda<float> d_normalize_params_;  // float[6]

    // 后处理输出（原始分辨率）
    unique_ptr_cuda<uchar>  d_buffer_dst_depth_;
    unique_ptr_cuda<uchar3> d_buffer_dst_colormap_;

    // 主机端 pinned memory（D2H 异步拷贝目标）
    unique_ptr_pinned_cuda<uchar>  host_pinned_depth_output_data_;
    unique_ptr_pinned_cuda<uchar3> host_pinned_depth_colormap_data_;

    // CPU 端资源
    cv::Mat colormap_table_;
};
