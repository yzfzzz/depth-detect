#include "depth_model.h"

#include "cub_utils.h"
#include "frame.h"
#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <cstring>
#include <fstream>
#include <memory>

DepthModel::DepthModel() {}

DepthModel::~DepthModel() {}

bool DepthModel::init(const std::string & model_path,
                      int                 raw_img_w,
                      int                 raw_img_h,
                      bool                is_normalize) {
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

    // 初始化 CUDA 资源（如果使用 TensorRT）
    if (backend_->getBackendType() == BackendType::TENSORRT) {
        // 分配 GPU 内存
        auto alloc_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
            return ptr;
        };

        d_buffer_[0].reset(alloc_cuda(3 * input_h_ * input_w_ * sizeof(float)));
        d_buffer_[1].reset(alloc_cuda(input_h_ * input_w_ * sizeof(float)));

        d_buffer_norm_depth_.reset(
            static_cast<uchar *>(alloc_cuda(input_h_ * input_w_ * sizeof(uchar))));
        d_buffer_norm_colormap_.reset(
            static_cast<uchar3 *>(alloc_cuda(input_h_ * input_w_ * sizeof(uchar3))));
        d_depth_infer_min_value_.reset(static_cast<float *>(alloc_cuda(sizeof(float))));
        d_depth_infer_max_value_.reset(static_cast<float *>(alloc_cuda(sizeof(float))));
        d_mean_.reset(static_cast<float *>(alloc_cuda(3 * sizeof(float))));
        d_std_.reset(static_cast<float *>(alloc_cuda(3 * sizeof(float))));
        d_before_preprocess_img_data_.reset(
            static_cast<uchar *>(alloc_cuda(3 * raw_img_h_ * raw_img_w_ * sizeof(uchar))));

        // 拷贝归一化参数到 GPU
        CHECK_CUDA(
            cudaMemcpy(d_mean_.get(), h_mean_.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(
            cudaMemcpy(d_std_.get(), h_std_.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

        // 查询 CUB 所需临时显存
        cub_get_min_max_temp_bytes(static_cast<float *>(d_buffer_[1].get()), input_h_ * input_w_,
                                   &cub_min_bytes_, &cub_max_bytes_, stream_);
        cub_bytes_ = std::max(cub_min_bytes_, cub_max_bytes_);
        d_cub_mid_min_.reset(alloc_cuda(cub_bytes_));
        d_cub_mid_max_.reset(alloc_cuda(cub_bytes_));

        h_output_data_.resize(input_h_ * input_w_);

        // 分配固定主机内存
        auto alloc_pinned_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMallocHost(&ptr, bytes));
            return ptr;
        };

        host_pinned_depth_colormap_data_.reset(
            static_cast<uchar3 *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar3))));
        host_pinned_depth_output_data_.reset(
            static_cast<uchar *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar))));

        // 分配目标缓冲区
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
    }

    APP_INFO("DepthModel initialized successfully with backend: {}",
             backendTypeToString(backend_->getBackendType()));
    return true;
}

// std::vector<float> DepthModel::preProcessCPU(const cv::Mat & image) {
//     // CPU 预处理（用于 ONNX Runtime）
//     cv::Mat resized_image, rgb;
//     cv::resize(image, resized_image, cv::Size(input_w_, input_h_));
//     cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);

//     std::vector<float> input_tensor;
//     input_tensor.reserve(3 * input_h_ * input_w_);

//     for (int k = 0; k < 3; k++) {
//         for (int i = 0; i < resized_image.rows; i++) {
//             for (int j = 0; j < resized_image.cols; j++) {
//                 float value = static_cast<float>(rgb.at<cv::Vec3b>(i, j)[k]) / 255.0f;
//                 input_tensor.push_back((value - h_mean_[k]) / h_std_[k]);
//             }
//         }
//     }

//     return input_tensor;
// }

bool DepthModel::runInference(FrameInputContext & frame_input_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return {};
    }

    // TODO: 实现同步推理逻辑
    APP_INFO("Running synchronous inference for Depth model...");
    return false;
}

// GPU 版本推理链路
void DepthModel::cudaPreProcess(FrameInputContext & frame_input_context) {
    depthPreprocess(frame_input_context.d_raw_img_.get(), (float *) d_buffer_[0].get(), raw_img_w_,
                    raw_img_h_, input_w_, input_h_, d_mean_.get(), d_std_.get(), stream_);
}

bool DepthModel::runInferenceAsync(FrameInputContext & frame_input_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }

    if (backend_->getBackendType() == BackendType::TENSORRT) {
        // 异步预处理
        preProcess(frame_input_context);
        // 异步推理
        backend_->runInferenceAsync(d_buffer_[0].get(), d_buffer_[1].get(), stream_);
        // 异步后处理
        postProcess(frame_input_context);
        return true;
    }
    APP_WARN("Async inference not supported for ONNX Runtime backend");
    return false;
}

void DepthModel::cudaPostProcess(FrameInputContext & frame_input_context) {
    // Use wrapper in op_kernel to perform reductions
    cub_device_reduce_min(d_cub_mid_min_.get(), cub_bytes_, (float *) d_buffer_[1].get(),
                          d_depth_infer_min_value_.get(), input_h_ * input_w_, stream_);
    cub_device_reduce_max(d_cub_mid_max_.get(), cub_bytes_, (float *) d_buffer_[1].get(),
                          d_depth_infer_max_value_.get(), input_h_ * input_w_, stream_);

    normalize_colormap_resize(
        (float *) d_buffer_[1].get(), d_buffer_norm_depth_.get(), d_buffer_norm_colormap_.get(),
        d_buffer_dst_depth_.get(), d_buffer_dst_colormap_.get(), d_depth_infer_min_value_.get(),
        d_depth_infer_max_value_.get(), input_w_, input_h_, raw_img_w_, raw_img_h_, stream_);

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


void DepthModel::cvMatPreProcess(FrameInputContext & frame_input_context) {
    APP_INFO("Running CPU pre-processing for Depth model...");
}

void DepthModel::cvMatPostProcess(FrameInputContext & frame_input_context) {
    APP_INFO("Running CPU post-processing for Depth model...");
}
