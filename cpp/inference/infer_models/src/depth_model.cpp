#include "depth_model.h"

#include "cub_utils.h"
#include "frame.h"
#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>

DepthModel::DepthModel() {}

DepthModel::~DepthModel() {}

bool DepthModel::init(std::map<std::string, std::string> model_path,
                      int                                raw_img_w,
                      int                                raw_img_h,
                      bool                               is_normalize) {
    raw_img_w_    = raw_img_w;
    raw_img_h_    = raw_img_h;
    is_normalize_ = is_normalize;

    if (raw_img_h_ <= 0 || raw_img_w_ <= 0) {
        APP_ERROR("Invalid image dimensions: {}x{}", raw_img_w_, raw_img_h_);
        return false;
    }

    // 设置归一化参数
    if (!is_normalize_) {
        h_mean_ = { 0.0f, 0.0f, 0.0f };
        h_std_  = { 1.0f, 1.0f, 1.0f };
    } else {
        h_mean_ = { 0.485f, 0.456f, 0.406f };
        h_std_  = { 0.229f, 0.224f, 0.225f };
    }

    // 设置模型路径
    model_path_ = model_path;

    // 初始化后端
    if (!initInferenceBackend()) {
        return false;
    }

    // 获取输入输出维度
    auto input_dims = getInputDims();
    input_h_        = input_dims[2];
    input_w_        = input_dims[3];

    // 初始化 CUDA 资源（仅 TensorRT 后端）
    if (backend_->getBackendType() == BackendType::TensorRT) {
        auto alloc_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
            return ptr;
        };

        // ── 推理 I/O 缓冲区 ──
        d_buffer_[0].reset(alloc_cuda(3 * input_h_ * input_w_ * sizeof(float)));
        d_buffer_[1].reset(alloc_cuda(input_h_ * input_w_ * sizeof(float)));

        // ── 后处理中间 buffer（模型分辨率）──
        d_buffer_norm_depth_.reset(
            static_cast<uchar *>(alloc_cuda(input_h_ * input_w_ * sizeof(uchar))));
        d_buffer_norm_colormap_.reset(
            static_cast<uchar3 *>(alloc_cuda(input_h_ * input_w_ * sizeof(uchar3))));

        // ── 归约标量：min + max 合并为 float[2] ──
        d_depth_minmax_.reset(static_cast<float *>(alloc_cuda(2 * sizeof(float))));

        // ── 预处理参数：mean[3] + std[3] 合并为 float[6]，一次拷贝到 GPU ──
        d_normalize_params_.reset(static_cast<float *>(alloc_cuda(6 * sizeof(float))));
        float h_params[6];
        std::memcpy(h_params, h_mean_.data(), 3 * sizeof(float));
        std::memcpy(h_params + 3, h_std_.data(), 3 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(d_normalize_params_.get(), h_params, 6 * sizeof(float),
                              cudaMemcpyHostToDevice));

        // ── CUB 归约临时空间（min/max 串行调用，复用同一 buffer）──
        {
            size_t min_bytes = 0, max_bytes = 0;
            cub_get_min_max_temp_bytes(static_cast<float *>(d_buffer_[1].get()),
                                       input_h_ * input_w_, &min_bytes, &max_bytes, stream_);
            cub_temp_bytes_ = std::max(min_bytes, max_bytes);
        }
        d_cub_temp_.reset(alloc_cuda(cub_temp_bytes_));

        h_output_data_.resize(input_h_ * input_w_);

        // ── 主机端 pinned memory ──
        auto alloc_pinned_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMallocHost(&ptr, bytes));
            return ptr;
        };

        host_pinned_depth_output_data_.reset(
            static_cast<uchar *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar))));
        host_pinned_depth_colormap_data_.reset(
            static_cast<uchar3 *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar3))));

        // ── 后处理输出 buffer（原始分辨率）──
#if defined(__aarch64__) && defined(ENABLE_JESTON_MEM_MANAGED)
        void * dst_depth    = nullptr;
        void * dst_colormap = nullptr;
        CHECK_CUDA(cudaMallocManaged(&dst_depth, raw_img_h_ * raw_img_w_ * sizeof(uchar)));
        CHECK_CUDA(cudaMallocManaged(&dst_colormap, raw_img_h_ * raw_img_w_ * sizeof(uchar3)));
        d_buffer_dst_depth_.reset(static_cast<uchar *>(dst_depth));
        d_buffer_dst_colormap_.reset(static_cast<uchar3 *>(dst_colormap));
#else
        d_buffer_dst_depth_.reset(
            static_cast<uchar *>(alloc_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar))));
        d_buffer_dst_colormap_.reset(
            static_cast<uchar3 *>(alloc_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar3))));
#endif

        // 初始化颜色映射表
        initColorMapTable();
    } else if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        auto output_size = getOutputSize();
        h_output_data_.resize(output_size / sizeof(float));
    }

    APP_INFO("DepthModel initialized successfully with backend: {}",
             backendTypeToString(backend_->getBackendType()));
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// GPU 推理链路 (TensorRT)
// ═══════════════════════════════════════════════════════════════════════

void DepthModel::cudaPreProcess(FrameInputContext & frame_input_context) {
    depthPreprocess(frame_input_context.d_raw_img_.get(), static_cast<float *>(d_buffer_[0].get()),
                    raw_img_w_, raw_img_h_, input_w_, input_h_,
                    d_normalize_params_.get(),      // mean
                    d_normalize_params_.get() + 3,  // std
                    stream_);
}

bool DepthModel::runInferenceAsync(FrameInputContext & frame_input_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }

    if (backend_->getBackendType() == BackendType::TensorRT) {
        cudaPreProcess(frame_input_context);
        backend_->runInferenceAsync(d_buffer_[0].get(), d_buffer_[1].get(), stream_);
        cudaPostProcess(frame_input_context);
        return true;
    }
    APP_WARN("Async inference not supported for ONNX Runtime backend");
    return false;
}

void DepthModel::cudaPostProcess(FrameInputContext & frame_input_context) {
    // CUB 归约：min / max（串行，复用 d_cub_temp_）
    cub_device_reduce_min(d_cub_temp_.get(), cub_temp_bytes_,
                          static_cast<float *>(d_buffer_[1].get()),
                          d_depth_minmax_.get(),  // d_out → min
                          input_h_ * input_w_, stream_);

    cub_device_reduce_max(d_cub_temp_.get(), cub_temp_bytes_,
                          static_cast<float *>(d_buffer_[1].get()),
                          d_depth_minmax_.get() + 1,  // d_out → max
                          input_h_ * input_w_, stream_);

    // 归一化 + 颜色映射 + resize
    normalize_colormap_resize(static_cast<float *>(d_buffer_[1].get()), d_buffer_norm_depth_.get(),
                              d_buffer_norm_colormap_.get(), d_buffer_dst_depth_.get(),
                              d_buffer_dst_colormap_.get(),
                              d_depth_minmax_.get(),      // min
                              d_depth_minmax_.get() + 1,  // max
                              input_w_, input_h_, raw_img_w_, raw_img_h_, stream_);

    // 异步 D2H 拷贝
    CHECK_CUDA(cudaMemcpyAsync(host_pinned_depth_output_data_.get(), d_buffer_dst_depth_.get(),
                               raw_img_h_ * raw_img_w_ * sizeof(uchar), cudaMemcpyDeviceToHost,
                               stream_));
    CHECK_CUDA(cudaMemcpyAsync(host_pinned_depth_colormap_data_.get(), d_buffer_dst_colormap_.get(),
                               raw_img_h_ * raw_img_w_ * sizeof(uchar3), cudaMemcpyDeviceToHost,
                               stream_));
}

void DepthModel::getInferOutputResult(InferOutputContext & infer_output_context) {
    waitAsync();
    infer_output_context.result_depth =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC1, host_pinned_depth_output_data_.get());
    infer_output_context.depth_vis =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC3, host_pinned_depth_colormap_data_.get());
}

// ═══════════════════════════════════════════════════════════════════════
// CPU 推理链路 (ONNX Runtime)
// ═══════════════════════════════════════════════════════════════════════

bool DepthModel::runInference(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }

    if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        cvMatPreProcess(frame_input_context);
        if (!backend_->runInference(onnx_input_tensor_.data(), h_output_data_.data())) {
            APP_ERROR("ONNX Runtime inference failed");
            return false;
        }
        cvMatPostProcess(infer_output_context);
        return true;
    }
    return false;
}

void DepthModel::cvMatPreProcess(FrameInputContext & frame_input_context) {
    cv::Mat resized_image, rgb;
    cv::resize(frame_input_context.raw_img, resized_image, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);

    onnx_input_tensor_.clear();
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < resized_image.rows; i++) {
            for (int j = 0; j < resized_image.cols; j++) {
                onnx_input_tensor_.emplace_back(
                    (static_cast<float>(rgb.at<cv::Vec3b>(i, j)[k]) / 255.0f - h_mean_[k]) /
                    h_std_[k]);
            }
        }
    }
}

void DepthModel::cvMatPostProcess(InferOutputContext & infer_output_context) {
    cv::Mat depth_mat(input_h_, input_w_, CV_32FC1, h_output_data_.data());
    cv::normalize(depth_mat, depth_mat, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat colormap;
    cv::applyColorMap(depth_mat, colormap, cv::COLORMAP_INFERNO);
    cv::resize(colormap, colormap, cv::Size(raw_img_w_, raw_img_h_));
    infer_output_context.result_depth = depth_mat;
    infer_output_context.depth_vis    = colormap;
}
