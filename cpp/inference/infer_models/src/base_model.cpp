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

void BaseModel::init(std::map<std::string, std::string> model_path, int raw_img_w, int raw_img_h) {
    model_path_ = model_path;
    raw_img_w_  = raw_img_w;
    raw_img_h_  = raw_img_h;
}

bool BaseModel::isGPUAvailable() {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

std::unique_ptr<InferenceBackend> BaseModel::createBackend() {
    // 优先尝试 TensorRT（如果 GPU 可用且用户偏好 GPU）
    if (isGPUAvailable()) {
        APP_INFO("GPU detected, attempting to use TensorRT backend...");
        auto it = model_path_.find("engine");
        if (it != model_path_.end()) {
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
    auto it = model_path_.find("onnx");
    if (it != model_path_.end()) {
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

bool BaseModel::initInferenceBackend() {
    if (initialized_) {
        APP_WARN("Model already initialized");
        return true;
    }

    // 检查模型文件是否存在
    for (auto & [type, path] : model_path_) {
        std::ifstream file(path);
        if (!file.good()) {
            APP_ERROR("Model file not found: type={}, path: {}", type, path);
            return false;
        }
        file.close();
    }

    // 创建后端
    backend_ = createBackend();
    if (!backend_) {
        APP_ERROR("Failed to create inference backend");
        return false;
    }

    initialized_ = true;
    APP_INFO("Model initialized successfully with backend: {}",
             backendTypeToString(backend_->getBackendType()));
    return true;
}
