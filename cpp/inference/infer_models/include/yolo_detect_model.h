#pragma once

#include "base_model.h"
#include "public.h"

#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    std::array<float, 4> bbox;  // x1, y1, x2, y2
    float                conf;
    int                  classId;
};

/**
 * @brief YOLO 目标检测模型
 * 
 * 继承自 BaseModel，支持 TensorRT 和 ONNX Runtime 后端
 */
class YoloDetectModel : public BaseModel {
  public:
    void init(const std::string & model_path,
              int                 raw_img_w,
              int                 raw_img_h,
              float               nms_thresh,
              float               conf_thresh,
              int                 num_class);

    /**
     * @brief 同步推理
     */
    std::vector<Detection> runInference(void * input_data, void * output_data);

    /**
     * @brief 异步推理
     */
    void                   runInferenceAsync(uchar * d_image);
    void                   waitAsync();
    std::vector<Detection> getInferResultAsync(const cv::Mat & img);

  private:
    void preProcess(const cv::Mat & input, void * output) override;
    void postProcess(void * output, void * results) override;

    void cudaPreProcess(uchar * input) override;
    void cudaPostProcess() override;

    /**
     * @brief 后处理检测结果
     */
    std::vector<Detection> postProcessDetections(const cv::Mat & img);

  private:
    int   num_class_;
    float nms_thresh_;
    float conf_thresh_;

    // CUDA 资源（仅 TensorRT 后端使用）
    std::array<unique_ptr_cuda<void>, 2> d_buffer_;
    unique_ptr_cuda<float>               d_transpose_;
    unique_ptr_cuda<float>               d_decode_;
    unique_ptr_cuda<uchar>               d_src_data_;
    unique_ptr_cuda<uchar>               d_mid_data_;
    unique_ptr_pinned_cuda<float>        h_output_data_;

    int output_candidates_;

    // 常量
    static constexpr int MAX_NUM_OUTPUT_BBOX = 1000;
    static constexpr int NUM_BOX_ELEMENT     = 7;
};
