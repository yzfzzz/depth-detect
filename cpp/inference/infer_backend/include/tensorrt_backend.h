#pragma once

#include "inference_backend.h"
#include "memory.h"
#include "logger_manager.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief TensorRT 推理后端（GPU 加速）
 */
class TensorRTBackend : public InferenceBackend {
public:
    TensorRTBackend(int gpu_id = 0);
    ~TensorRTBackend() override;

    // InferenceBackend 接口实现
    bool loadModel(const std::string& model_path) override;
    bool runInference(void* input_data, void* output_data) override;
    bool runInferenceAsync(void* input_data, void* output_data, cudaStream_t stream) override;
    std::vector<int> getInputDims() const override;
    std::vector<int> getOutputDims() const override;
    size_t getInputSize() const override;
    size_t getOutputSize() const override;
    BackendType getBackendType() const override { return BackendType::TENSORRT; }
    bool isAvailable() const override;

    // TensorRT 特有方法
    nvinfer1::IExecutionContext* getContext() const { return context_.get(); }

private:
    bool loadEngine(const std::string& engine_path);
    void setupInputOutputDims();

private:
    int gpu_id_;
    Logger logger_;

    // TensorRT 组件
    TrtRuntimePtr runtime_;
    TrtEnginePtr engine_;
    TrtContextPtr context_;
    // 维度信息
    std::vector<int> input_dims_;
    std::vector<int> output_dims_;
    size_t input_size_;
    size_t output_size_;

    // Tensor 名称
    std::string input_tensor_name_;
    std::string output_tensor_name_;
};