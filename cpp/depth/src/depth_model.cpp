#include "depth_model.h"

#include "cub_utils.h"
#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <memory.h>
#include <opencv2/core/hal/interface.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

DepthModel::DepthModel() : stream_(0), input_h_(0), input_w_(0) {}

DepthModel::~DepthModel() {
    context_.reset();
    engine_.reset();
    runtime_.reset();

    // 确保所有在 stream_ 上的 CUDA 工作完成
    if (stream_ != 0) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = 0;
    }
}

void DepthModel::init(const std::string & engine_path, int img_w, int img_h, bool is_normalize) {
    raw_img_h_ = img_h;
    raw_img_w_ = img_w;
    assert(raw_img_h_ > 0 && raw_img_w_ > 0 && "Invalid image dimensions");

    if (!is_normalize) {
        h_mean_ = { 0, 0, 0 };
        h_std_  = { 1.0f, 1.0f, 1.0f };
    } else {
        h_mean_ = { 0.485f, 0.456f, 0.406f };
        h_std_  = { 0.229f, 0.224f, 0.225f };
    }

    std::ifstream engineStream(engine_path, std::ios::binary);
    if (!engineStream.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    engineStream.seekg(0, std::ios::end);
    size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engineData.get(), modelSize));
    context_.reset(engine_->createExecutionContext());

    // 兼容 TensorRT 8.x
#if NV_TENSORRT_MAJOR < 10
    int nb_bindings = engine_->getNbBindings();
    assert(nb_bindings == 2 && "Expecting exactly 2 bindings (input and output)");
    io_tensor_name_[0] = engine_->getBindingName(0);
    io_tensor_name_[1] = engine_->getBindingName(1);
    // Define input dimensions
    auto input_dims    = engine_->getBindingDimensions(0);
    input_h_           = input_dims.d[2];
    input_w_           = input_dims.d[3];

#else
    int nb_bindings = engine_->getNbIOTensors();
    assert(nb_bindings == 2 && "Expecting exactly 2 bindings (input and output)");
    io_tensor_name_[0] = engine_->getIOTensorName(0);
    io_tensor_name_[1] = engine_->getIOTensorName(1);
    auto input_dims    = engine_->getTensorShape(io_tensor_name_[0].c_str());
    input_h_           = input_dims.d[2];
    input_w_           = input_dims.d[3];
#endif

    cudaStreamCreate(&stream_);

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

    CHECK_CUDA(
        cudaMemcpy(d_mean_.get(), h_mean_.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_std_.get(), h_std_.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

    d_before_preprocess_img_data_.reset(
        static_cast<uchar *>(alloc_cuda(3 * raw_img_h_ * raw_img_w_ * sizeof(uchar))));

    // 查询 CUB 所需临时显存大小（由封装函数实现）
    cub_get_min_max_temp_bytes(static_cast<float *>(d_buffer_[1].get()), input_h_ * input_w_,
                               &cub_min_bytes_, &cub_max_bytes_, stream_);
    cub_bytes_ = std::max(cub_min_bytes_, cub_max_bytes_);
    d_cub_mid_min_.reset(alloc_cuda(cub_bytes_));
    d_cub_mid_max_.reset(alloc_cuda(cub_bytes_));
    h_output_data_.resize(input_h_ * input_w_);

#if NV_TENSORRT_MAJOR >= 10
    // TRT 8.x: 不需要显式 setTensorAddress，enqueueV2 会按 binding 顺序读取 buffer 数组
    // TRT 10.x: 必须显式设置 Tensor 地址
    context_->setTensorAddress(io_tensor_name_[0].c_str(), d_buffer_[0].get());
    context_->setTensorAddress(io_tensor_name_[1].c_str(), d_buffer_[1].get());
#endif

    auto alloc_pinned_cuda = [](size_t bytes) {
        void * ptr = nullptr;
        CHECK_CUDA(cudaMallocHost(&ptr, bytes));
        return ptr;
    };
    host_pinned_depth_colormap_data_.reset(
        static_cast<uchar3 *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar3))));

    host_pinned_depth_output_data_.reset(
        static_cast<uchar *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar))));
#if defined(__aarch64__) && defined(ENABLE_JESTON_MEM_MANAGED)
    // Jeston 上使用统一内存
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

    initColorMapTable();
}

std::vector<float> DepthModel::preProcess(const cv::Mat & image) {
    raw_img_w_ = image.cols;
    raw_img_h_ = image.rows;
    // 自定义的 resize_depth (假设你在其他地方定义了它)
    cv::Mat resized_image, rgb;  // std::get<0>(resize_depth(image, input_w, input_h));
    cv::resize(image, resized_image, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input_tensor;

    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < resized_image.rows; i++) {
            for (int j = 0; j < resized_image.cols; j++) {
                input_tensor.emplace_back(
                    ((float) rgb.at<cv::Vec3b>(i, j)[k] / 255.0f - h_mean_[k]) / h_std_[k]);
            }
        }
    }
    return input_tensor;
}

std::pair<cv::Mat, cv::Mat> DepthModel::predict(const cv::Mat & image) {
    std::vector<float> input = preProcess(image);
    CHECK_CUDA(cudaMemcpy(d_buffer_[0].get(), input.data(), 3 * input_h_ * input_w_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    void * buffer_ptrs[2] = { d_buffer_[0].get(), d_buffer_[1].get() };
    bool   status         = context_->executeV2(buffer_ptrs);
    if (!status) {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return {};
    }

    CHECK_CUDA(cudaMemcpy(h_output_data_.data(), d_buffer_[1].get(),
                          input_h_ * input_w_ * sizeof(float), cudaMemcpyDeviceToHost));
    cv::Mat depth_mat(input_h_, input_w_, CV_32FC1, h_output_data_.data());
    cv::normalize(depth_mat, depth_mat, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat colormap;
    cv::applyColorMap(depth_mat, colormap, cv::COLORMAP_INFERNO);
    cv::resize(colormap, colormap, cv::Size(raw_img_w_, raw_img_h_));
    return std::make_pair(depth_mat, colormap);
}

void DepthModel::predictAsync(uchar * d_image) {
    // 数据异步拷贝至 GPU, 并进行 cuda 前处理
    {
        nvtx3::scoped_range tracker_scope("depth preprocess");
        preProcessAsync(d_image);
    }
    {
        nvtx3::scoped_range tracker_scope("depth enqueue");
#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
        // For Jetson Nano (ARM64) and older TensorRT versions
        void * buffer_ptrs[2] = { d_buffer_[0].get(), d_buffer_[1].get() };
        bool   status         = context_->enqueueV2(buffer_ptrs, stream_, nullptr);

#else
        bool status = context_->enqueueV3(stream_);
#endif
        if (!status) {
            std::cerr << "TensorRT enqueue failed!" << std::endl;
            return;
        }
    }

    {
        nvtx3::scoped_range tracker_scope("depth post process");
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
        CHECK_CUDA(cudaMemcpyAsync(
            host_pinned_depth_colormap_data_.get(), d_buffer_dst_colormap_.get(),
            raw_img_h_ * raw_img_w_ * sizeof(uchar3), cudaMemcpyDeviceToHost, stream_));
    }
}

void DepthModel::waitAsync() {
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

std::pair<cv::Mat, cv::Mat> DepthModel::getPredictResultAsync() {
    auto depth_output =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC1, host_pinned_depth_output_data_.get());
    auto depth_colormap =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC3, host_pinned_depth_colormap_data_.get());
    std::pair<cv::Mat, cv::Mat> result = std::make_pair(depth_output, depth_colormap);
    return result;
}

void DepthModel::preProcessAsync(uchar * d_image) {
    depthPreprocess(d_image, (float *) d_buffer_[0].get(), raw_img_w_, raw_img_h_, input_w_,
                    input_h_, d_mean_.get(), d_std_.get(), stream_);
}
