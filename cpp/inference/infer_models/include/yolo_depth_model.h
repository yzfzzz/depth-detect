#pragma once

#include "base_model.h"
#include "frame.h"

#include <map>
#include <opencv2/opencv.hpp>
#include <vector>

// YOLO 深度估计模型，输出单通道 dense depth map。
class YoloDepthModel : public BaseModel {
  public:
    bool init(std::map<std::string, std::string> model_path,
              int                                raw_img_w,
              int                                raw_img_h,
              bool                               use_gpu           = true,
              const std::string &                preferred_backend = "auto");

    void getInferOutputResult(InferOutputContext & infer_output_context) override;

    std::vector<float> cvMatPreProcess(FrameInputContext & frame_input_context) override;
    void               cvMatPostProcess(InferOutputContext & infer_output_context) override;
    void               cudaPreProcess(FrameInputContext & frame_input_context) override;
    void               cudaPostProcess(FrameInputContext & frame_input_context) override;

  private:
    // ONNX/CPU 推理路径的可视化回退（P1/P99 + TURBO，逻辑同 CUDA 版）
    void postProcessDepth(const std::vector<float> & depth,
                          InferOutputContext &       infer_output_context) const;
    void buildDepthVisualization(const cv::Mat &      model_depth,
                                 InferOutputContext & infer_output_context) const;

    // CUDA 伪彩色输出缓冲（原始分辨率，cudaPostProcess 写入 / getInferOutputResult 读取）
    unique_ptr_device<uchar>       d_buffer_dst_depth_;     // 归一化灰度 8U
    unique_ptr_device<uchar3>      d_buffer_dst_colormap_;  // TURBO 伪彩 BGR
    unique_ptr_pinned_host<uchar>  host_pinned_depth_output_data_;
    unique_ptr_pinned_host<uchar3> host_pinned_depth_colormap_data_;

    //  P1/P99 归一化的设备端中间缓冲（每帧复用，由 floatDepthColormapResize 消费）
    unique_ptr_device<float> d_stage_float_;  // 原始分辨率 float 深度 [raw_h*raw_w]
    unique_ptr_device<float> d_stat_min_;     // [DEPTH_COLORMAP_STAT_BLOCKS]
    unique_ptr_device<float> d_stat_max_;     // [DEPTH_COLORMAP_STAT_BLOCKS]
    unique_ptr_device<int>   d_stat_count_;   // [DEPTH_COLORMAP_STAT_BLOCKS]
    unique_ptr_device<float> d_range_;        // [2] {min, max}
    unique_ptr_device<int>   d_hist_;         // [256]
    unique_ptr_device<float> d_percentiles_;  // [2] {P1, P99}
};
