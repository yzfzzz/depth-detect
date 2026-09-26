#pragma once

#include "inference_backend.h"
#include "logger_manager.h"
#include "memory.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <string>
#include <vector>

// TensorRT 推理后端（GPU 加速）
class TensorRTBackend : public InferenceBackend {
  public:
    TensorRTBackend(int gpu_id = 0);
    ~TensorRTBackend() override;

    // InferenceBackend 接口实现
    bool loadModel(const std::string & model_path) override;

    // 单输出的便捷重载由基类提供
    using InferenceBackend::runInference;
    using InferenceBackend::runInferenceAsync;

    bool runInference(void * input_data, std::vector<void *> output_data) override;
    bool runInferenceAsync(void *              input_data,
                           std::vector<void *> output_data,
                           cudaStream_t        stream) override;

    // 异步能力查询：返回空串表示支持真异步（enqueueV2/enqueueV3 提交即返回）
    std::string asyncUnsupportedReason() const override { return {}; }

    std::vector<int>     getInputDims() const override;
    std::vector<int64_t> getOutputDims(int output_index = 0) const override;
    size_t               getInputByteSize() const override;
    size_t               getOutputByteSize(int output_index = 0) const override;

    BackendType getBackendType() const override { return BackendType::TensorRT; }

    std::string getBackendTypeName() const override { return "TensorRT"; }

    bool isAvailable() const override;

    // TensorRT 特有方法
    nvinfer1::IExecutionContext * getContext() const { return context_.get(); }

    virtual int getOutputIndexFromName(const std::string & name) const override {
        for (int i = 0; i < output_tensor_.size(); ++i) {
            if (output_tensor_[i].name == name) {
                return i + 1;
            }
        }
        return -1;  // 返回 -1 表示未找到
    }

  private:
    bool loadEngine(const std::string & engine_path);
    void setupInputOutputDims();

  private:
    int    gpu_id_;
    Logger logger_;

    // TensorRT 组件
    TrtRuntimePtr    runtime_;
    TrtEnginePtr     engine_;
    TrtContextPtr    context_;
    // 输入维度信息, Tensor 名称
    std::vector<int> input_dims_;
    size_t           input_byte_size_;
    std::string      input_tensor_name_;
};
