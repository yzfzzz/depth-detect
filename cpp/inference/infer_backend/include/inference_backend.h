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

// 推理后端抽象接口，支持多种推理后端：TensorRT (GPU)、ONNX Runtime (CPU/GPU)
class InferenceBackend {
  public:
    virtual ~InferenceBackend() = default;

    // 加载模型
    virtual bool loadModel(const std::string & model_path) = 0;

    // 同步：返回时输出已就绪
    virtual bool runInference(void * input_data, std::vector<void *> output_data) = 0;

    // 异步：把工作提交到 stream 后立即返回，输出的就绪顺序由 stream 保证
    virtual bool runInferenceAsync(void *              input_data,
                                   std::vector<void *> output_data,
                                   cudaStream_t        stream) = 0;

    virtual std::string asyncUnsupportedReason() const = 0;

    bool isAsyncInferenceSupported() const { return asyncUnsupportedReason().empty(); }

    bool runInference(void * input_data, void * output_data) {
        return runInference(input_data, std::vector<void *>{ output_data });
    }

    bool runInferenceAsync(void * input_data, void * output_data, cudaStream_t stream) {
        return runInferenceAsync(input_data, std::vector<void *>{ output_data }, stream);
    }

    // 获取输入维度
    virtual std::vector<int> getInputDims() const = 0;

    // 获取输出维度
    virtual std::vector<int64_t> getOutputDims(int output_index = 0) const = 0;

    // 获取输入数据大小（字节）
    virtual size_t getInputByteSize() const = 0;

    // 获取输出数据大小（字节）
    virtual size_t getOutputByteSize(int output_index = 0) const = 0;

    // 获取后端类型
    virtual BackendType getBackendType() const     = 0;
    virtual std::string getBackendTypeName() const = 0;

    // 检查后端是否可用
    virtual bool isAvailable() const = 0;

    // 获取输出个数
    virtual size_t getNumOutputs() const { return num_outputs_; }

    // 根据输出名获取输出索引
    virtual int getOutputIndexFromName(const std::string & name) const = 0;

  protected:
    // 异步降级策略（「查能力 → 记录查询结果与原因 → 回落同步」）的唯一实现点，
    bool degradeToSyncInference(void * input_data, std::vector<void *> output_data);

    struct OutputTensorInfo {
        std::string          name;
        std::vector<int64_t> dims;
        size_t               byte_size;
    };

    // 输出维度信息, Tensor 名称
    std::vector<OutputTensorInfo> output_tensor_;

    int num_outputs_ = 1;
};
