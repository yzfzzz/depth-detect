#include "depth_model.h"

#include "frame.h"
#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <cstring>
#include <map>
#include <memory>
#include <string>

bool DepthModel::init(std::map<std::string, std::string> model_path,
                      int                                raw_img_w,
                      int                                raw_img_h,
                      bool                               is_normalize,
                      bool                               use_gpu) {
    BaseModel::init(model_path, raw_img_w, raw_img_h, use_gpu);
    is_normalize_ = is_normalize;

    // 设置归一化参数：不归一化时保持原值 [0,255]，归一化时使用 ImageNet 标准均值/标准差
    if (!is_normalize_) {
        h_mean_ = { 0.0f, 0.0f, 0.0f };
        h_std_  = { 1.0f, 1.0f, 1.0f };
    } else {
        h_mean_ = { 0.485f, 0.456f, 0.406f };
        h_std_  = { 0.229f, 0.224f, 0.225f };
    }

    // 初始化 CUDA 资源（仅 TensorRT 后端）
    if (backend_->getBackendType() == BackendType::TensorRT) {
        // 封装 cudaMalloc 为返回裸指针的 lambda，配合 unique_ptr_cuda 自动管理显存生命周期
        auto alloc_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
            return ptr;
        };

        // 推理 I/O 缓冲区：d_infer_io_[0]=输入，d_infer_io_[1..N]=各输出，按索引对应 TRT binding
        d_infer_io_.resize(getNumOutputs() + 1);  // 输入 + 输出
        d_infer_io_[0].reset(alloc_cuda(getInputByteSize()));
        for (int i = 0; i < getNumOutputs(); ++i) {
            d_infer_io_[i + 1].reset(alloc_cuda(getOutputByteSize(i)));
        }

        // 后处理中间 buffer：模型分辨率下的归一化深度图和颜色映射图
        d_buffer_norm_depth_.reset(
            static_cast<uchar *>(alloc_cuda(input_h_ * input_w_ * sizeof(uchar))));
        d_buffer_norm_colormap_.reset(
            static_cast<uchar3 *>(alloc_cuda(input_h_ * input_w_ * sizeof(uchar3))));

        // 预处理参数：mean[3]+std[3] 合并为 float[6]，一次 H2D 拷贝到设备常量内存
        d_normalize_params_.reset(static_cast<float *>(alloc_cuda(6 * sizeof(float))));
        float h_params[6];
        std::memcpy(h_params, h_mean_.data(), 3 * sizeof(float));
        std::memcpy(h_params + 3, h_std_.data(), 3 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(d_normalize_params_.get(), h_params, 6 * sizeof(float),
                              cudaMemcpyHostToDevice));

        // 主机端 pinned memory：作为 D2H 异步拷贝的目标，避免同步等待
        // 封装 cudaMallocHost，分配页锁定内存用于异步 D2H 拷贝，避免 cudaMemcpy 阻塞 CPU
        auto alloc_pinned_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMallocHost(&ptr, bytes));
            return ptr;
        };

        host_pinned_depth_output_data_.reset(
            static_cast<uchar *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar))));
        host_pinned_depth_colormap_data_.reset(
            static_cast<uchar3 *>(alloc_pinned_cuda(raw_img_h_ * raw_img_w_ * sizeof(uchar3))));

        // 后处理输出 buffer（原始分辨率）
        // Jetson 统一内存架构下使用 cudaMallocManaged 避免显式拷贝；x86 平台用设备内存 + pinned memory D2H
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
        int output_num = getNumOutputs();
        h_infer_out_.resize(output_num);
        for (int i = 0; i < output_num; ++i) {
            auto output_size = getOutputByteSize(i);
            APP_INFO("ONNX Runtime output {} byte size: {} bytes", i, output_size);
            h_infer_out_[i].resize(output_size / sizeof(float));
        }
    }

    APP_INFO("DepthModel initialized successfully");
    return true;
}

// GPU 推理链路 (TensorRT) - 预处理、后处理均在 GPU 侧执行
void DepthModel::cudaPreProcess(FrameInputContext & frame_input_context) {
    if (frame_input_context.d_raw_img_ == nullptr) {
        APP_ERROR("Input image buffer is not allocated on GPU");
        return;
    }
    depthPreprocess(frame_input_context.d_raw_img_.get(),
                    static_cast<float *>(d_infer_io_[0].get()), raw_img_w_, raw_img_h_, input_w_,
                    input_h_,
                    d_normalize_params_.get(),      // mean
                    d_normalize_params_.get() + 3,  // std
                    stream_);
}

void DepthModel::cudaPostProcess(FrameInputContext & frame_input_context) {
    // 深度归一化 + 颜色映射 + resize 到原始分辨率
    normalize_colormap_resize(
        static_cast<float *>(d_infer_io_[getOutputIndexFromName("disp_output")].get()),
        d_buffer_norm_depth_.get(), d_buffer_norm_colormap_.get(), d_buffer_dst_depth_.get(),
        d_buffer_dst_colormap_.get(), input_w_, input_h_, raw_img_w_, raw_img_h_, stream_);

    // 异步 D2H 拷贝
    CHECK_CUDA(cudaMemcpyAsync(host_pinned_depth_output_data_.get(), d_buffer_dst_depth_.get(),
                               raw_img_h_ * raw_img_w_ * sizeof(uchar), cudaMemcpyDeviceToHost,
                               stream_));
    CHECK_CUDA(cudaMemcpyAsync(host_pinned_depth_colormap_data_.get(), d_buffer_dst_colormap_.get(),
                               raw_img_h_ * raw_img_w_ * sizeof(uchar3), cudaMemcpyDeviceToHost,
                               stream_));
}

void DepthModel::getInferOutputResult(InferOutputContext & infer_output_context) {
    // 同步等待所有异步操作完成（预处理→推理→后处理→D2H拷贝），然后读取结果
    synchronizeStream();
    infer_output_context.depth_raw_infer_out.resize(input_h_ * input_w_);
    cudaMemcpy(infer_output_context.depth_raw_infer_out.data(),
               d_infer_io_[getOutputIndexFromName("disp_output")].get(),
               input_h_ * input_w_ * sizeof(float), cudaMemcpyDeviceToHost);
    infer_output_context.result_depth =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC1, host_pinned_depth_output_data_.get());
    infer_output_context.depth_vis =
        cv::Mat(raw_img_h_, raw_img_w_, CV_8UC3, host_pinned_depth_colormap_data_.get());
}

std::vector<float> DepthModel::cvMatPreProcess(FrameInputContext & frame_input_context) {
    cv::Mat resized_image, rgb;
    cv::resize(frame_input_context.raw_img, resized_image, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);
    std::vector<float> onnx_input_tensor;
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < resized_image.rows; i++) {
            for (int j = 0; j < resized_image.cols; j++) {
                onnx_input_tensor.emplace_back(
                    (static_cast<float>(rgb.at<cv::Vec3b>(i, j)[k]) / 255.0f - h_mean_[k]) /
                    h_std_[k]);
            }
        }
    }
    return onnx_input_tensor;
}

void DepthModel::cvMatPostProcess(InferOutputContext & infer_output_context) {
    infer_output_context.depth_raw_infer_out = h_infer_out_[getOutputIndexFromName("disp_output")];
    cv::Mat depth_mat(input_h_, input_w_, CV_32FC1,
                      h_infer_out_[getOutputIndexFromName("disp_output")].data());
    cv::normalize(depth_mat, depth_mat, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat colormap;
    // 归一化到 [0,255] 后用 INFERNO 色谱着色（近=亮黄，远=深紫黑）
    cv::applyColorMap(depth_mat, colormap, cv::COLORMAP_INFERNO);
    cv::resize(colormap, colormap, cv::Size(raw_img_w_, raw_img_h_));
    infer_output_context.result_depth = depth_mat;
    infer_output_context.depth_vis    = colormap;
}
