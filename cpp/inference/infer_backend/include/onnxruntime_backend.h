#pragma once

#include "inference_backend.h"
#include "logger_manager.h"

#include <onnxruntime_cxx_api.h>

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
    std::vector<int> getInputDims() const override;
    std::vector<int> getOutputDims() const override;
    size_t           getInputSize() const override;
    size_t           getOutputSize() const override;

    BackendType getBackendType() const override { return BackendType::OnnxRuntime; }

    bool isAvailable() const override;

  private:
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::Session                         session_;
    std::unique_ptr<Ort::MemoryInfo>     memory_info_;

    std::vector<int64_t> input_dims_;
    std::vector<int64_t> output_dims_;
    size_t               input_size_;
    size_t               output_size_;

    std::string input_name_;
    std::string output_name_;
};
