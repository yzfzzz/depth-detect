#include "yolo_depth_model.h"

#include "logger_manager.h"
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
        auto alloc_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
            return ptr;
        };

        d_infer_io_.resize(2);
        d_infer_io_[0].reset(alloc_cuda(getInputByteSize()));
        d_infer_io_[1].reset(alloc_cuda(getOutputByteSize(0)));
        // 中间数据缓冲区（预处理后的图像数据）
        d_mid_data_.reset(
            static_cast<uchar *>(alloc_cuda(sizeof(uchar) * input_h_ * input_w_ * 3)));
        // 深度→伪彩色改在 CPU 侧完成（buildDepthVisualization，与 Python 参考一致），
        // 因此这里不再需要 norm/dst/pinned 颜色缓冲。
    } else if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        h_infer_out_.resize(1);
        h_infer_out_[0].resize(getOutputByteSize(0) / sizeof(float));
    }
    APP_INFO("YoloDepthModel initialized successfully");
    return true;
}

void YoloDepthModel::cudaPreProcess(FrameInputContext & frame_input_context) {
    APP_INFO("YoloDepthModel cudaPreProcess");
    if (frame_input_context.d_raw_img_ == nullptr) {
        APP_ERROR("Input image buffer is not allocated on GPU");
        return;
    }

    preprocess_v2(static_cast<float *>(d_infer_io_[0].get()), frame_input_context.d_raw_img_.get(),
                  d_mid_data_.get(), raw_img_h_, raw_img_w_, input_h_, input_w_, stream_);
}

void YoloDepthModel::cudaPostProcess(FrameInputContext &) {
    APP_INFO("YoloDepthModel cudaPostProcess");
    // 伪彩色可视化已改到 CPU 侧完成（getInferOutputResult -> buildDepthVisualization），
    // 与 Python 参考实现（去 letterbox -> resize -> P1/P99 归一化 -> TURBO）逐像素一致，
    // GPU 侧无需再做 normalize_colormap_resize + D2H。
}

void YoloDepthModel::getInferOutputResult(InferOutputContext & infer_output_context) {
    APP_INFO("YoloDepthModel getInferOutputResult");
    // 同步等待所有异步操作完成（预处理→推理→后处理→D2H拷贝），然后读取结果
    synchronizeStream();
    APP_INFO("YoloDepthModel getInferOutputResult synchronizeStream done");
    infer_output_context.depth_raw_infer_out.resize(input_h_ * input_w_);
    cudaMemcpy(infer_output_context.depth_raw_infer_out.data(),
               d_infer_io_[getOutputIndexFromName("output0")].get(),
               input_h_ * input_w_ * sizeof(float), cudaMemcpyDeviceToHost);

    // 用原始 float 深度（模型分辨率，含 letterbox pad）在 CPU 侧生成 result_depth / depth_vis，
    // 流程与 Python 参考实现 depth_to_colormap 一致
    const cv::Mat model_depth(input_h_, input_w_, CV_32FC1,
                              infer_output_context.depth_raw_infer_out.data());
    buildDepthVisualization(model_depth, infer_output_context);

    // ===== DEBUG: 打印 result_depth 内容（统计值 + 8x8 采样网格） =====
    const cv::Mat & depth = infer_output_context.result_depth;
    if (depth.empty()) {
        APP_INFO("result_depth is EMPTY ({}x{})", raw_img_w_, raw_img_h_);
    } else {
        double min_v = 0.0, max_v = 0.0;
        cv::minMaxLoc(depth, &min_v, &max_v);
        const double mean_v = cv::mean(depth)[0];
        APP_INFO("result_depth {}x{} CV_8UC1: min={} max={} mean={:.1f}", depth.cols, depth.rows,
                 static_cast<int>(min_v), static_cast<int>(max_v), mean_v);

        // 采样网格：行/列各取 8 个采样点（含首尾），打印每个点的灰度值
        const int rows = depth.rows;
        const int cols = depth.cols;
        APP_INFO("result_depth sampled grid (rows={} cols={}):", rows, cols);
        for (int gr = 0; gr < 8; ++gr) {
            const int r = gr == 7 ? rows - 1 : (gr * rows) / 8;
            std::ostringstream line;
            for (int gc = 0; gc < 8; ++gc) {
                const int c = gc == 7 ? cols - 1 : (gc * cols) / 8;
                line << std::setw(4) << static_cast<int>(depth.at<uchar>(r, c)) << " ";
            }
            APP_INFO("  row {:>4}: {}", r, line.str());
        }
    }
}

std::vector<float> YoloDepthModel::cvMatPreProcess(FrameInputContext & frame_input_context) {
    const cv::Mat & image = frame_input_context.raw_img;
    const float scale = std::min(static_cast<float>(input_w_) / image.cols,
                                 static_cast<float>(input_h_) / image.rows);
    const int resized_w = static_cast<int>(image.cols * scale);
    const int resized_h = static_cast<int>(image.rows * scale);
    const int pad_left = (input_w_ - resized_w) / 2;
    const int pad_top = (input_h_ - resized_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, input_h_ - resized_h - pad_top, pad_left,
                       input_w_ - resized_w - pad_left, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    std::vector<float> tensor(3 * input_h_ * input_w_);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_h_; ++y) {
            const uchar * row = padded.ptr<uchar>(y);
            for (int x = 0; x < input_w_; ++x) {
                tensor[c * input_h_ * input_w_ + y * input_w_ + x] =
                    row[x * 3 + (2 - c)] / 255.0f;
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

    // 与 getInferOutputResult(TensorRT 路径) 共用同一套 CPU 可视化逻辑
    const cv::Mat model_depth(input_h_, input_w_, CV_32FC1,
                              infer_output_context.depth_raw_infer_out.data());
    buildDepthVisualization(model_depth, infer_output_context);
}

void YoloDepthModel::buildDepthVisualization(const cv::Mat &   model_depth,
                                             InferOutputContext & infer_output_context) const {
    // 对应 Python 参考实现：model.infer() 内部的 remove_letterbox + depth_to_colormap
    // 1) 去掉 letterbox 灰边：几何与 preprocess_v2/cvMatPreProcess 一致（截断取整、居中）
    const float scale = std::min(static_cast<float>(input_w_) / raw_img_w_,
                                 static_cast<float>(input_h_) / raw_img_h_);
    const int   resized_w = static_cast<int>(raw_img_w_ * scale);
    const int   resized_h = static_cast<int>(raw_img_h_ * scale);
    const int   pad_left  = (input_w_ - resized_w) / 2;
    const int   pad_top   = (input_h_ - resized_h) / 2;

    const cv::Mat cropped =
        model_depth(cv::Rect(pad_left, pad_top, resized_w, resized_h)).clone();

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