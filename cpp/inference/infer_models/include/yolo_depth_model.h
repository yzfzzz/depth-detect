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
              bool                               use_gpu = true);

    void getInferOutputResult(InferOutputContext & infer_output_context) override;

    std::vector<float> cvMatPreProcess(FrameInputContext & frame_input_context) override;
    void               cvMatPostProcess(InferOutputContext & infer_output_context) override;
    void               cudaPreProcess(FrameInputContext & frame_input_context) override;
    void               cudaPostProcess(FrameInputContext & frame_input_context) override;

  private:
    void postProcessDepth(const std::vector<float> & depth,
                          InferOutputContext &       infer_output_context) const;

    // 把模型分辨率(含 letterbox)的 float 深度图，按 Python 参考实现的 depth_to_colormap
    // 流程（去 pad → resize 回原图 → P1/P99 分位数归一化 → clip → applyColorMap）转成
    // result_depth(灰度 CV_8UC1) 与 depth_vis(TURBO 伪彩 CV_8UC3)
    void buildDepthVisualization(const cv::Mat & model_depth,
                                 InferOutputContext & infer_output_context) const;

    unique_ptr_cuda<uchar> d_mid_data_;
};
