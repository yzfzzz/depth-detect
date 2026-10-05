#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum class BackendType {
    TensorRT,     // TensorRT(GPU)
    OnnxRuntime,  // ONNX Runtime(CPU)
    Unknown,
};

// 模型张量的驻留位置，决定模型层走哪条预处理/后处理路径：
//   CudaDevice -> 设备缓冲 + CUDA kernel 前后处理（cudaPreProcess/cudaPostProcess）
//   Host       -> 主机缓冲 + OpenCV 前后处理（cvMatPreProcess/cvMatPostProcess）
// 新增后端只需如实上报自己的 TensorLocation，模型层据此自动选择路径，
enum class TensorLocation {
    Host,
    CudaDevice,
};

// 推理后端抽象接口，支持多种推理后端：TensorRT (GPU)、ONNX Runtime (CPU/GPU)、QNN (NPU)。
// 本接口不包含任何 SDK 头文件（CUDA/TensorRT/ONNX Runtime/QNN），
// 具体类型一律以 void* / 原生句柄透传，由各后端实现自行解释
class InferenceBackend {
  public:
    virtual ~InferenceBackend() = default;

    // 加载模型
    virtual bool loadModel(const std::string & model_path) = 0;

    // 同步：返回时输出已就绪
    virtual bool runInference(void * input_data, std::vector<void *> output_data) = 0;

    // 异步：把工作提交后立即返回，输出的就绪顺序由 stream_handle 指向的原生流对象保证。
    // stream_handle 是后端相关的原生流句柄（CUDA 后端解释为 cudaStream_t，
    // Host 后端可忽略该参数）；仅支持同步的后端实现 asyncUnsupportedReason 说明原因
    virtual bool runInferenceAsync(void *              input_data,
                                   std::vector<void *> output_data,
                                   void *              stream_handle) = 0;

    virtual std::string asyncUnsupportedReason() const = 0;

    bool isAsyncInferenceSupported() const { return asyncUnsupportedReason().empty(); }

    bool runInference(void * input_data, void * output_data) {
        return runInference(input_data, std::vector<void *>{ output_data });
    }

    bool runInferenceAsync(void * input_data, void * output_data, void * stream_handle) {
        return runInferenceAsync(input_data, std::vector<void *>{ output_data }, stream_handle);
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

    // 张量驻留位置（模型层据此选择预处理/后处理路径）
    virtual TensorLocation getTensorLocation() const = 0;

    // 检查后端是否可用
    virtual bool isAvailable() const = 0;

    // 获取输出个数
    virtual size_t getNumOutputs() const { return num_outputs_; }

    // 根据输出名获取输出索引（统一语义：0-based 输出序号，未找到返回 -1）。
    // CUDA 路径的缓冲槽位布局为 [input, out0, out1, ...]，模型侧取输出缓冲时需 +1 偏移
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
