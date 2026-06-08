#include "base_model.h"

#include "onnxruntime_backend.h"
#include "tensorrt_backend.h"

#include <cuda_runtime_api.h>

#include <fstream>

BaseModel::BaseModel() : initialized_(false), stream_(0) {}

BaseModel::~BaseModel() {
    if (stream_ != 0) {
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        CHECK_CUDA(cudaStreamDestroy(stream_));
    }
}

void BaseModel::init(std::map<std::string, std::string> model_path,
                     int                                raw_img_w,
                     int                                raw_img_h,
                     bool                               use_gpu) {
    if (raw_img_h <= 0 || raw_img_w <= 0) {
        APP_ERROR("Invalid image dimensions: {}x{}", raw_img_w, raw_img_h);
        return;
    }
    raw_img_w_  = raw_img_w;
    raw_img_h_  = raw_img_h;
    bool status = initInferenceBackend(model_path, use_gpu);
    if (!status) {
        APP_ERROR("Failed to initialize inference backend");
        return;
    }
    // 获取输入输出维度
    auto input_dims  = getInputDims();
    auto output_dims = getOutputDims();
    // 设置输入尺寸
    input_h_         = input_dims[2];
    input_w_         = input_dims[3];

    APP_INFO("Model input size: {}x{}", input_w_, input_h_);
}

bool BaseModel::isGPUAvailable() {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

std::unique_ptr<InferenceBackend> BaseModel::createBackend(
    std::map<std::string, std::string> model_path,
    bool                               use_gpu) {
    // 优先尝试 TensorRT（如果 GPU 可用且用户偏好 GPU）
    APP_INFO("Checking for GPU availability: {}", isGPUAvailable() ? "Yes" : "No");
    APP_INFO("GPU preference: {}", use_gpu ? "Yes" : "No");
    if (isGPUAvailable() && use_gpu) {
        auto it = model_path.find("engine");
        if (it != model_path.end()) {
            // TODO: 需要自动检测gpu id
            auto trt_backend = std::make_unique<TensorRTBackend>(0);
            if (trt_backend->loadModel(it->second)) {
                APP_INFO("TensorRT backend initialized successfully");
                // 创建 CUDA 流
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaStreamCreate(&stream_));
                APP_INFO("CUDA stream created successfully");
                return trt_backend;
            }
        }
        APP_WARN("TensorRT engine path not found, falling back to ONNX Runtime");
    }
    auto it = model_path.find("onnx");
    if (it != model_path.end()) {
        auto onnx_backend = std::make_unique<OnnxRuntimeBackend>();
        if (onnx_backend->loadModel(it->second)) {
            APP_INFO("ONNX Runtime CPU backend initialized successfully");
            return onnx_backend;
        }
    }
    APP_ERROR("ONNX Runtime CPU backend path not found in {}, failed to initialize any backend",
              it->second);
    return nullptr;
}

bool BaseModel::initInferenceBackend(std::map<std::string, std::string> model_path, bool use_gpu) {
    if (initialized_) {
        APP_WARN("Model already initialized");
        return true;
    }

    // 检查模型文件是否存在
    for (auto & item : model_path) {
        const auto &  type = item.first;
        const auto &  path = item.second;
        std::ifstream file(path);
        if (!file.good()) {
            APP_ERROR("Model file not found: type={}, path: {}", type, path);
            return false;
        }
        file.close();
    }

    // 创建后端
    backend_ = createBackend(model_path, use_gpu);
    if (!backend_) {
        APP_ERROR("Failed to create inference backend");
        return false;
    }

    initialized_ = true;
    APP_INFO("Model initialized successfully with backend: {}",
             backendTypeToString(backend_->getBackendType()));
    return true;
}

// 执行异步推理（供子类调用）
bool BaseModel::runInferenceAsync(FrameInputContext & frame_input_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }
    if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        APP_ERROR(
            "Asynchronous inference not supported for ONNX Runtime backend, use runInference "
            "instead");
        return false;
    } else if (backend_->getBackendType() == BackendType::TensorRT) {
        // 异步预处理
        cudaPreProcess(frame_input_context);
        // 异步推理
        backend_->runInferenceAsync(d_infer_io_[0].get(), d_infer_io_[1].get(), stream_);
        // 异步后处理
        cudaPostProcess(frame_input_context);
        return true;
    }
    return false;
}

// 执行同步推理（供子类调用）
bool BaseModel::runInference(FrameInputContext &  frame_input_context,
                             InferOutputContext & infer_output_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }
    if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        std::vector<float> onnx_input_tensor = cvMatPreProcess(frame_input_context);
        cudaStreamSynchronize(stream_);  // 确保预处理完成
        backend_->runInference(onnx_input_tensor.data(), h_infer_out_.data());
        cvMatPostProcess(infer_output_context);
        return true;
    } else if (backend_->getBackendType() == BackendType::TensorRT) {
        // 异步预处理
        cudaPreProcess(frame_input_context);
        synchronizeStream();  // 等待预处理完成
        // 同步推理
        backend_->runInference(d_infer_io_[0].get(), d_infer_io_[1].get());
        // 异步后处理
        cudaPostProcess(frame_input_context);
        getInferOutputResult(infer_output_context);
        return true;
    }
    return false;
}
