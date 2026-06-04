#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum class BackendType {
    TENSORRT,  // TensorRT(GPU)
    ONNX_CPU,  // ONNX Runtime(CPU)
    NONE,
};

static inline std::string backendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::TENSORRT:
            return "TensorRT";
        case BackendType::ONNX_CPU:
            return "ONNX Runtime (CPU)";
        default:
            return "Unknown";
    }
}

// 推理后端抽象接口，支持多种推理后端：TensorRT (GPU)、ONNX Runtime (CPU/GPU)
class InferenceBackend {
  public:
    virtual ~InferenceBackend() = default;

    // 加载模型
    virtual bool loadModel(const std::string & model_path) = 0;

    // 执行推理
    virtual bool runInference(void * input_data, void * output_data) = 0;

    // 执行异步推理
    virtual bool runInferenceAsync(void * input_data, void * output_data, cudaStream_t stream) = 0;

    // 获取输入维度
    virtual std::vector<int> getInputDims() const = 0;

    // 获取输出维度
    virtual std::vector<int> getOutputDims() const = 0;

    // 获取输入数据大小（字节）
    virtual size_t getInputSize() const = 0;

    // 获取输出数据大小（字节）
    virtual size_t getOutputSize() const = 0;

    // 获取后端类型
    virtual BackendType getBackendType() const = 0;

    // 检查后端是否可用
    virtual bool isAvailable() const = 0;
};
