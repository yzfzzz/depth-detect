#include "yolo_depth_model.h"

#include "logger_manager.h"
#include "postprocess.h"
#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

// 与 numpy.percentile(v, p*100, method='linear') 一致的线性插值分位数。
// 会原地重排 v；只收集有限值后调用，v 非空。
double percentileLinear(std::vector<float> & v, double p) {
    const size_t n = v.size();
    if (n == 0) {
        return 0.0;
    }
    if (n == 1) {
        return v[0];
    }
    const double pos = (static_cast<double>(n) - 1.0) * p;
    const size_t lo  = static_cast<size_t>(pos);  // floor(pos)，pos >= 0
    const size_t hi  = std::min(lo + 1, n - 1);
    const double w   = pos - static_cast<double>(lo);

    // 找到第 lo 小元素：nth_element 保证 [0, lo) <= v[lo] <= [lo, n)
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(lo), v.end());
    const double a = v[lo];
    if (w == 0.0) {
        return a;
    }
    // 第 lo+1 小元素 = 右半区 [lo+1, n) 的最小值
    const double b = *std::min_element(v.begin() + static_cast<std::ptrdiff_t>(hi), v.end());
    return a * (1.0 - w) + b * w;
}

}  // namespace

bool YoloDepthModel::init(std::map<std::string, std::string> model_path,
                          int                                raw_img_w,
                          int                                raw_img_h,
                          bool                               use_gpu) {
    BaseModel::init(model_path, raw_img_w, raw_img_h, use_gpu);
    if (!isBackendInitialized()) {
        return false;
    }

    if (getNumOutputs() != 1) {
        APP_ERROR("YoloDepthModel expects one output, got {}", getNumOutputs());
        return false;
    }

    const auto output_dims = getOutputDims(0);
    if (output_dims.size() != 4 || output_dims[1] != 1 || output_dims[2] != input_h_ ||
        output_dims[3] != input_w_) {
        APP_ERROR("Unexpected YOLO depth output shape");
        return false;
    }

    if (backend_->getBackendType() == BackendType::TensorRT) {
        d_infer_io_.resize(2);
        d_infer_io_[0].reset(allocCuda(getInputByteSize()));
        d_infer_io_[1].reset(allocCuda(getOutputByteSize(0)));
        // 中间数据缓冲区（预处理后的图像数据）
        d_mid_data_.reset(static_cast<uchar *>(allocCuda(sizeof(uchar) * getInputHxW() * 3)));

        // CUDA 伪彩色输出缓冲（原始分辨率）
        d_buffer_dst_depth_.reset(static_cast<uchar *>(allocCuda(getRawImgHxW() * sizeof(uchar))));
        d_buffer_dst_colormap_.reset(
            static_cast<uchar3 *>(allocCuda(getRawImgHxW() * sizeof(uchar3))));
        host_pinned_depth_output_data_.reset(
            static_cast<uchar *>(allocPinnedCuda(getRawImgHxW() * sizeof(uchar))));
        host_pinned_depth_colormap_data_.reset(
            static_cast<uchar3 *>(allocPinnedCuda(getRawImgHxW() * sizeof(uchar3))));

        // P1/P99 归一化中间缓冲（floatDepthColormapResize 每帧复用）
        const size_t stat_bytes = DEPTH_COLORMAP_STAT_BLOCKS * sizeof(float);
        d_stage_float_.reset(static_cast<float *>(allocCuda(getRawImgHxW() * sizeof(float))));
        d_stat_min_.reset(static_cast<float *>(allocCuda(stat_bytes)));
        d_stat_max_.reset(static_cast<float *>(allocCuda(stat_bytes)));
        d_stat_count_.reset(
            static_cast<int *>(allocCuda(DEPTH_COLORMAP_STAT_BLOCKS * sizeof(int))));
        d_range_.reset(static_cast<float *>(allocCuda(2 * sizeof(float))));
        d_hist_.reset(static_cast<int *>(allocCuda(256 * sizeof(int))));
        d_percentiles_.reset(static_cast<float *>(allocCuda(2 * sizeof(float))));

        // TURBO 颜色表上传到 __constant__ 内存（cv::applyColorMap 生成，与 Python 一致）
        initTurboColorTable();
    } else if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        h_infer_out_.resize(1);
        h_infer_out_[0].resize(getOutputByteSize(0) / sizeof(float));
    }
    APP_INFO("YoloDepthModel initialized successfully");
    return true;
}

void YoloDepthModel::cudaPreProcess(FrameInputContext & frame_input_context) {
    if (frame_input_context.d_raw_img_ == nullptr) {
        APP_ERROR("Input image buffer is not allocated on GPU");
        return;
    }

    preprocess_v2(static_cast<float *>(d_infer_io_[0].get()), frame_input_context.d_raw_img_.get(),
                  d_mid_data_.get(), raw_img_h_, raw_img_w_, input_h_, input_w_, stream_);
}

void YoloDepthModel::cudaPostProcess(FrameInputContext &) {
    // 在 stream 上依次执行 resize、P1/P99 分位数统计、归一化 + TURBO 查表，
    // 全程异步、无主机同步；结果异步 D2H 到 pinned 内存，由 getInferOutputResult 读取。

    // letterbox 内容区（去掉灰边），几何与 preprocess_v2 一致（截断取整、居中）
    const float scale = std::min(static_cast<float>(input_w_) / raw_img_w_,
                                 static_cast<float>(input_h_) / raw_img_h_);
    const int   roi_w = static_cast<int>(raw_img_w_ * scale);
    const int   roi_h = static_cast<int>(raw_img_h_ * scale);
    const int   roi_x = (input_w_ - roi_w) / 2;
    const int   roi_y = (input_h_ - roi_h) / 2;

    floatDepthColormapResize(
        static_cast<float *>(d_infer_io_[getOutputIndexFromName("output0")].get()), input_w_, roi_x,
        roi_y, roi_w, roi_h, static_cast<float *>(d_stage_float_.get()), raw_img_w_, raw_img_h_,
        static_cast<uchar *>(d_buffer_dst_depth_.get()),
        static_cast<uchar3 *>(d_buffer_dst_colormap_.get()), d_stat_min_.get(), d_stat_max_.get(),
        d_stat_count_.get(), d_range_.get(), d_hist_.get(), d_percentiles_.get(),
        DEPTH_COLORMAP_STAT_BLOCKS, stream_);

    // 异步 D2H 拷贝（与上面 kernel 同 stream，getInferOutputResult 同步后即可读）

    CHECK_CUDA(cudaMemcpyAsync(host_pinned_depth_output_data_.get(), d_buffer_dst_depth_.get(),
                               getRawImgHxW() * sizeof(uchar), cudaMemcpyDeviceToHost,
                               stream_));
    CHECK_CUDA(cudaMemcpyAsync(host_pinned_depth_colormap_data_.get(), d_buffer_dst_colormap_.get(),
                               getRawImgHxW() * sizeof(uchar3), cudaMemcpyDeviceToHost,
                               stream_));
}

void YoloDepthModel::getInferOutputResult(InferOutputContext & infer_output_context) {
    // 同步等待所有异步操作完成（预处理→推理→后处理→D2H拷贝），然后读取结果
    synchronizeStream();

    infer_output_context.depth_raw_infer_out.resize(getInputHxW());
    cudaMemcpy(infer_output_context.depth_raw_infer_out.data(),
               d_infer_io_[getOutputIndexFromName("output0")].get(),
               getInputHxW() * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaPostProcess 已把归一化灰度 / TURBO 伪彩异步拷到 pinned 内存，
    infer_output_context.result_depth =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC1, host_pinned_depth_output_data_.get());
    infer_output_context.depth_vis =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC3, host_pinned_depth_colormap_data_.get());
}

std::vector<float> YoloDepthModel::cvMatPreProcess(FrameInputContext & frame_input_context) {
    const cv::Mat & image     = frame_input_context.raw_img;
    const float     scale     = std::min(static_cast<float>(input_w_) / image.cols,
                                         static_cast<float>(input_h_) / image.rows);
    const int       resized_w = static_cast<int>(image.cols * scale);
    const int       resized_h = static_cast<int>(image.rows * scale);
    const int       pad_left  = (input_w_ - resized_w) / 2;
    const int       pad_top   = (input_h_ - resized_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, input_h_ - resized_h - pad_top, pad_left,
                       input_w_ - resized_w - pad_left, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    std::vector<float> tensor(3 * getInputHxW());
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_h_; ++y) {
            const uchar * row = padded.ptr<uchar>(y);
            for (int x = 0; x < input_w_; ++x) {
                tensor[c * getInputHxW() + y * input_w_ + x] = row[x * 3 + (2 - c)] / 255.0f;
            }
        }
    }
    return tensor;
}

void YoloDepthModel::cvMatPostProcess(InferOutputContext & infer_output_context) {
    postProcessDepth(h_infer_out_[getOutputIndexFromName("output0")], infer_output_context);
}

void YoloDepthModel::postProcessDepth(const std::vector<float> & depth,
                                      InferOutputContext &       infer_output_context) const {
    const size_t expected_size = static_cast<size_t>(input_h_) * input_w_;
    if (depth.size() < expected_size) {
        APP_ERROR("YOLO depth output is too small: {} values, expected {}", depth.size(),
                  expected_size);
        return;
    }

    infer_output_context.depth_raw_infer_out.assign(depth.begin(), depth.begin() + expected_size);

    // ONNX/CPU 回退路径：TensorRT 路径的伪彩由 cudaPostProcess 在 GPU 完成，
    // 这里用 CPU 实现同等的 P1/P99 + TURBO 逻辑，保证两条路径输出一致。
    const cv::Mat model_depth(input_h_, input_w_, CV_32FC1,
                              infer_output_context.depth_raw_infer_out.data());
    buildDepthVisualization(model_depth, infer_output_context);
}

void YoloDepthModel::buildDepthVisualization(const cv::Mat &      model_depth,
                                             InferOutputContext & infer_output_context) const {
    // 对应 Python 参考实现：model.infer() 内部的 remove_letterbox + depth_to_colormap
    // 1) 去掉 letterbox 灰边：几何与 preprocess_v2/cvMatPreProcess 一致（截断取整、居中）
    const float scale     = std::min(static_cast<float>(input_w_) / raw_img_w_,
                                     static_cast<float>(input_h_) / raw_img_h_);
    const int   resized_w = static_cast<int>(raw_img_w_ * scale);
    const int   resized_h = static_cast<int>(raw_img_h_ * scale);
    const int   pad_left  = (input_w_ - resized_w) / 2;
    const int   pad_top   = (input_h_ - resized_h) / 2;

    const cv::Mat cropped = model_depth(cv::Rect(pad_left, pad_top, resized_w, resized_h)).clone();

    // 2) 缩放回原始分辨率（Python: cv2.resize INTER_LINEAR）
    cv::Mat depth_raw;
    cv::resize(cropped, depth_raw, cv::Size(raw_img_w_, raw_img_h_), 0, 0, cv::INTER_LINEAR);

    // 3) Python depth_to_colormap：只统计有限值，vmin=P1 / vmax=P99（对离群点鲁棒）
    std::vector<float> finite;
    finite.reserve(static_cast<size_t>(depth_raw.total()));
    for (int y = 0; y < depth_raw.rows; ++y) {
        const float * row = depth_raw.ptr<float>(y);
        for (int x = 0; x < depth_raw.cols; ++x) {
            if (std::isfinite(row[x])) {
                finite.push_back(row[x]);
            }
        }
    }

    cv::Mat gray;  // CV_8UC1，原始分辨率，Python 中的 gray = clip(normalized)*255
    if (finite.empty()) {
        gray = cv::Mat(raw_img_h_, raw_img_w_, CV_8UC1, cv::Scalar(0));
    } else {
        const double vmin  = percentileLinear(finite, 0.01);
        const double vmax  = percentileLinear(finite, 0.99);
        double       range = vmax - vmin;
        if (range <= 0.0) {  // Python: vmax <= vmin 时 vmax = vmin + 1e-6
            range = 1e-6;
        }
        // Python: gray = clip((d - vmin) / range) * 255（先归一化到 [0,1]，再放大 255）
        // convertTo 的饱和转换等价于 clip；四舍五入与 numpy 截断仅差 ±1
        cv::Mat normalized;
        depth_raw.convertTo(normalized, CV_32F, 255.0 / range, -255.0 * vmin / range);
        normalized.convertTo(gray, CV_8U);
    }

    infer_output_context.result_depth = gray.clone();
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 2)
    cv::applyColorMap(gray, infer_output_context.depth_vis, cv::COLORMAP_TURBO);
#else
    // 旧版 OpenCV 没有 COLORMAP_TURBO 时退回 JET，观感仍接近 Python turbo
    cv::applyColorMap(gray, infer_output_context.depth_vis, cv::COLORMAP_JET);
#endif
}
