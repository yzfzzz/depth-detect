#ifndef INFER_H
#define INFER_H

#include "memory.h"
#include "public.h"

#include <opencv2/core/hal/interface.h>

#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    // x1, y1, x2, y2
    std::array<float, 4> bbox;
    float                conf;
    int                  classId;
};

class YoloDetectModel {
  public:
    YoloDetectModel(const std::string trtFile,
                    int               raw_img_w,
                    int               raw_img_h,
                    int               gpuId      = 0,
                    float             nmsThresh  = 0.45f,
                    float             confThresh = 0.25f,
                    int               numClass   = 80);
    ~YoloDetectModel();
    std::vector<Detection> inference(const cv::Mat & img);
    void                   inferenceAsync(uchar * d_image);
    std::vector<Detection> postProcess(const cv::Mat & img);

    void waitAsync();

    std::vector<Detection> getInferResultAsync(const cv::Mat & img);

  private:
    void getEngine();

  private:
    Logger      g_logger_;
    std::string trtFile_;

    int   numClass_;
    float nmsThresh_;
    float confThresh_;

    TrtEnginePtr  engine_;
    TrtRuntimePtr runtime_;
    TrtContextPtr context_;

    cudaStream_t stream_;

    unique_ptr_pinned_cuda<float>        h_output_data_;
    std::array<unique_ptr_cuda<void>, 2> d_buffer_;
    unique_ptr_cuda<float>               d_transpose_;
    unique_ptr_cuda<float>               d_decode_;
    // preprocess
    unique_ptr_cuda<uchar>               d_src_data_;
    unique_ptr_cuda<uchar>               d_mid_data_;

    int  OUTPUT_CANDIDATES_;           // 8400: 80 * 80 + 40 * 40 + 20 * 20
    int  yolo26_max_num_output_bbox_;  // 暂时用于yolo26，后续可以删除
    int  yolo26_num_box_element_;      // 暂时用于yolo26，后续可以删除
    bool is_need_nms_ = true;

    int input_h_;
    int input_w_;
    int raw_img_h_;
    int raw_img_w_;

    // assume the box outputs no more than MAX_NUM_OUTPUT_BBOX boxes that conf >= NMS_THRESH;
    const int MAX_NUM_OUTPUT_BBOX = 1000;
    const int NUM_BOX_ELEMENT     = 7;
};

#endif  // INFER_H
