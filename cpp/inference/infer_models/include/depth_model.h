#pragma once

#include "base_model.h"
#include "memory.h"
#include "public.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 深度估计模型
 * 
 * 继承自 BaseModel，支持 TensorRT 和 ONNX Runtime 后端
 */
class DepthModel : public BaseModel {
  public:
    DepthModel();
    ~DepthModel() override;

    /**
     * @brief 初始化模型
     */
    bool init(const std::string & model_path, int raw_img_w, int raw_img_h, bool is_normalize);

    /**
     * @brief 同步推理
     */
    std::pair<cv::Mat, cv::Mat> runInference(void * input_data, void * output_data);

    /**
     * @brief 异步推理
     */
    void                        runInferenceAsync(uchar * d_image);
    void                        waitAsync();
    std::pair<cv::Mat, cv::Mat> getPredictResultAsync();

  private:
    // BaseModel 接口实现
    void preProcess(const cv::Mat & input, void * output) override;
    void postProcess(void * output, void * results) override;
    void cudaPreProcess(uchar * input) override;
    void cudaPostProcess() override;

    /**
     * @brief CPU 预处理（用于 ONNX Runtime）
     */
    std::vector<float> preProcessCPU(const cv::Mat & image);

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
