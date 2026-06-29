#pragma once

#include "inference_backend.h"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ONNX Runtime CPU 推理后端
class OnnxRuntimeBackend : public InferenceBackend {
  public:
    OnnxRuntimeBackend();
    ~OnnxRuntimeBackend() override;

    // InferenceBackend 接口实现
    bool loadModel(const std::string & model_path) override;
    bool runInference(void * input_data, void * output_data) override;
    bool runInferenceAsync(void * input_data, void * output_data, cudaStream_t stream) override;
    bool runInference(void * input_data, std::vector<void *> output_data) override;
    bool runInferenceAsync(void *              input_data,
                           std::vector<void *> output_data,
                           cudaStream_t        stream) override;
    std::vector<int>     getInputDims() const override;
    std::vector<int64_t> getOutputDims(int output_index = 0) const override;
    size_t               getInputByteSize() const override;
    size_t               getOutputByteSize(int output_index = 0) const override;

    BackendType getBackendType() const override { return BackendType::OnnxRuntime; }

    std::string getBackendTypeName() const override { return "ONNX Runtime (CPU)"; };

    bool isAvailable() const override;

    virtual size_t getOutputIndexFromName(const std::string & name) const override {
        for (size_t i = 0; i < output_tensor_.size(); ++i) {
            if (output_tensor_[i].name == name) {
                return i;
            }
        }
        return static_cast<size_t>(-1);  // 返回 -1 表示未找到
    }

  private:
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::Session                         session_;
    std::unique_ptr<Ort::MemoryInfo>     memory_info_;

    std::vector<int64_t> input_dims_;
    size_t               input_byte_size_;

    std::string input_name_;
};
