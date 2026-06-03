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

/**
 * @brief 推理后端抽象接口
 * 
 * 支持多种推理后端：TensorRT (GPU)、ONNX Runtime (CPU/GPU)
 */
class InferenceBackend {
  public:
    virtual ~InferenceBackend() = default;

    /**
     * @brief 加载模型
     * @param model_path 模型文件路径
     * @return 是否加载成功
     */
    virtual bool loadModel(const std::string & model_path) = 0;

    /**
     * @brief 执行推理
     * @param input_data 输入数据指针
     * @param output_data 输出数据指针
     * @return 是否推理成功
     */
    virtual bool runInference(void * input_data, void * output_data) = 0;

    /**
     * @brief 执行异步推理
     * @param input_data 输入数据指针
     * @param output_data 输出数据指针
     * @param stream CUDA 流
     * @return 是否推理成功
     */
    virtual bool runInferenceAsync(void * input_data, void * output_data, cudaStream_t stream) = 0;

    /**
     * @brief 获取输入维度
     * @return 输入维度 [batch, channels, height, width]
     */
    virtual std::vector<int> getInputDims() const = 0;

    /**
     * @brief 获取输出维度
     * @return 输出维度
     */
    virtual std::vector<int> getOutputDims() const = 0;

    /**
     * @brief 获取输入数据大小（字节）
     */
    virtual size_t getInputSize() const = 0;

    /**
     * @brief 获取输出数据大小（字节）
     */
    virtual size_t getOutputSize() const = 0;

    /**
     * @brief 获取后端类型
     */
    virtual BackendType getBackendType() const = 0;


    /**
     * @brief 检查后端是否可用
     */
    virtual bool isAvailable() const = 0;
};
