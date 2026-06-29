#include "base_model.h"

#include "logger_manager.h"
#include "onnxruntime_backend.h"
#include "tensorrt_backend.h"

#include <cuda_runtime_api.h>

#include <fstream>

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
    initialized_ = false;
    stream_      = 0;
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
    // 从后端获取模型要求的输入尺寸（如 640x640），用于后续预处理 resize
    auto input_dims  = getInputDims();
    auto output_dims = getOutputDims();
    input_h_         = input_dims[2];
    input_w_         = input_dims[3];
}

bool BaseModel::isGPUAvailable() {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

std::unique_ptr<InferenceBackend> BaseModel::createBackend(
    std::map<std::string, std::string> model_path,
    bool                               use_gpu) {
    // 后端选择策略：GPU 可用 + 用户偏好 GPU → 优先 TensorRT（.engine），兜底 ONNX（.onnx）
    // 仅 CPU 模式 → 直接走 ONNX Runtime
    APP_INFO("Checking for GPU availability: {}", isGPUAvailable() ? "Yes" : "No");
    APP_INFO("GPU preference: {}", use_gpu ? "Yes" : "No");
    if (use_gpu && isGPUAvailable()) {
        auto it = model_path.find("engine");
        if (it != model_path.end()) {
            // 创建 TRT 后端，默认 device=0（暂时不支持多 GPU 场景）
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
        APP_ERROR("ONNX Runtime CPU backend path not found in {}, failed to initialize any backend",
                  it->second);
    }
    APP_ERROR("Config yaml without onnx key, failed to initialize any backend");
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
    APP_INFO("Model initialized successfully with backend: {}", backend_->getBackendTypeName());
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
        // TensorRT 异步路径：预处理→推理→后处理全在 GPU Stream 上排队，CPU 不等待
        cudaPreProcess(frame_input_context);
        std::vector<void *> output_buffers;
        output_buffers.reserve(d_infer_io_.size() - 1);
        std::transform(d_infer_io_.begin() + 1, d_infer_io_.end(),
                       std::back_inserter(output_buffers), [](auto & ptr) { return ptr.get(); });
        backend_->runInferenceAsync(d_infer_io_[0].get(), output_buffers, stream_);
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
    std::vector<void *> output_buffers;
    output_buffers.reserve(getNumOutputs());
    if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        std::vector<float> onnx_input_tensor = cvMatPreProcess(frame_input_context);
        std::transform(h_infer_out_.begin(), h_infer_out_.end(), std::back_inserter(output_buffers),
                       [](auto & v) { return v.data(); });
        backend_->runInference(onnx_input_tensor.data(), output_buffers);
        cvMatPostProcess(infer_output_context);
        return true;
    } else if (backend_->getBackendType() == BackendType::TensorRT) {
        // TensorRT 同步路径：异步预处理 → 等待完成 → 同步推理 → 异步后处理 → 取结果
        cudaPreProcess(frame_input_context);
        synchronizeStream();  // 确保预处理数据就绪后再推理
        std::transform(d_infer_io_.begin() + 1, d_infer_io_.end(),
                       std::back_inserter(output_buffers), [](auto & ptr) { return ptr.get(); });
        backend_->runInference(d_infer_io_[0].get(), output_buffers);
        // 异步后处理
        cudaPostProcess(frame_input_context);
        getInferOutputResult(infer_output_context);
        return true;
    }
    return false;
}
