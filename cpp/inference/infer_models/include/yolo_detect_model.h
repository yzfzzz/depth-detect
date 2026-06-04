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

// YOLO 目标检测模型，继承自 BaseModel，支持 TensorRT 和 ONNX Runtime 后端
class YoloDetectModel : public BaseModel {
  public:
    void init(const std::string & model_path,
              int                 raw_img_w,
              int                 raw_img_h,
              float               nms_thresh,
              float               conf_thresh,
              int                 num_class);

    // 同步推理
    bool runInference(FrameInputContext &  frame_input_context,
                      InferOutputContext & infer_output_context) override;

    // 异步推理
    bool runInferenceAsync(FrameInputContext &  frame_input_context,
                           InferOutputContext & infer_output_context) override;
    void waitAsync();

  private:
    // BaseModel 接口实现
    // opencv 预处理和后处理（用于 ONNX Runtime）
    void cvMatPreProcess(FrameInputContext & frame_input_context) override;
    void cvMatPostProcess(FrameInputContext &  frame_input_context,
                          InferOutputContext & infer_output_context) override;
    // CUDA 预处理和后处理（用于 TensorRT）
    void cudaPreProcess(FrameInputContext & frame_input_context) override;
    void cudaPostProcess(FrameInputContext &  frame_input_context,
                         InferOutputContext & infer_output_context) override;


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

  public:
      

    // 常量
    static constexpr int MAX_NUM_OUTPUT_BBOX = 1000;
    static constexpr int NUM_BOX_ELEMENT     = 7;
};
