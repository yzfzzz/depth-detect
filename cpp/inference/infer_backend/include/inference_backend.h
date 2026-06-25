#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum class BackendType {
    TensorRT,     // TensorRT(GPU)
    OnnxRuntime,  // ONNX Runtime(CPU)
    Unkown,
};

static inline std::string backendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::TensorRT:
            return "TensorRT";
        case BackendType::OnnxRuntime:
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
    virtual bool runInference(void * input_data, void * output_data)              = 0;
    virtual bool runInference(void * input_data, std::vector<void *> output_data) = 0;

    // 执行异步推理
    virtual bool runInferenceAsync(void * input_data, void * output_data, cudaStream_t stream) = 0;
    virtual bool runInferenceAsync(void *              input_data,
                                   std::vector<void *> output_data,
                                   cudaStream_t        stream)                                        = 0;

    // 获取输入维度
    virtual std::vector<int> getInputDims() const = 0;

    // 获取输出维度
    virtual std::vector<int64_t> getOutputDims(int output_index = 0) const = 0;

    // 获取输入数据大小（字节）
    virtual size_t getInputByteSize() const = 0;

    // 获取输出数据大小（字节）
    virtual size_t getOutputByteSize(int output_index = 0) const = 0;

    // 获取后端类型
    virtual BackendType getBackendType() const = 0;

    // 检查后端是否可用
    virtual bool isAvailable() const = 0;

    void * getOutputData(size_t index) const;

    // 获取输出个数
    virtual size_t getNumOutputs() const { return num_outputs_; }

    // 根据输出名获取输出索引
    virtual size_t getOutputIndexFromName(const std::string & name) const {
        for (size_t i = 0; i < output_tensor_.size(); ++i) {
            if (output_tensor_[i].name == name) {
                if (getBackendType() == BackendType::OnnxRuntime) {
                    return i;      // ONNX Runtime 后端输出索引从 0 开始
                } else if (getBackendType() == BackendType::TensorRT) {
                    return i + 1;  // TensorRT 后端输出索引从 1 开始, 0是输入
                }
            }
        }
        return static_cast<size_t>(-1);  // 返回 -1 表示未找到
    }

  protected:
    struct OutputTensorInfo {
        std::string          name;
        std::vector<int64_t> dims;
        size_t               byte_size;
    };

    // 输出维度信息, Tensor 名称
    std::vector<OutputTensorInfo> output_tensor_;

    int num_outputs_ = 1;
};
