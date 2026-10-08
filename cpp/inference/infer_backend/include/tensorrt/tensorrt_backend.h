#pragma once

#include "cuda_utils.h"
#include "inference_backend.h"
#include "logger_manager.h"
#include "trt_logger.h"
#include "trt_memory.h"

#include <NvInfer.h>

#include <string>
#include <vector>

namespace inference {

// TensorRT 推理后端（GPU 加速）
class TensorRTBackend : public InferenceBackend {
  public:
    TensorRTBackend(int gpu_id = 0);
    ~TensorRTBackend() override;

    // InferenceBackend 接口实现
    bool loadModel(const std::string &        model_path,
                   const BackendInitOptions & options = BackendInitOptions()) override;

    // 单输出的便捷重载由基类提供
    using InferenceBackend::runInference;
    using InferenceBackend::runInferenceAsync;

    bool runInference(void * input_data, std::vector<void *> output_data) override;
    bool runInferenceAsync(void *              input_data,
                           std::vector<void *> output_data,
                           void *              stream_handle) override;

    // 异步能力查询：返回空串表示支持真异步（enqueueV2/enqueueV3 提交即返回）
    std::string asyncUnsupportedReason() const override { return {}; }

    std::vector<int>     getInputDims() const override;
    std::vector<int64_t> getOutputDims(int output_index = 0) const override;
    size_t               getInputByteSize() const override;
    size_t               getOutputByteSize(int output_index = 0) const override;

    BackendType getBackendType() const override { return BackendType::TensorRT; }

    std::string getBackendTypeName() const override { return "TensorRT"; }

    TensorLocation getTensorLocation() const override { return TensorLocation::CudaDevice; }

    bool isAvailable() const override;

    // TensorRT 特有方法
    nvinfer1::IExecutionContext * getContext() const { return context_.get(); }

    // 统一语义：0-based 输出序号（binding 序号 = 输出序号 + 1，输入占 binding 0）
    int getOutputIndexFromName(const std::string & name) const override {
        for (size_t i = 0; i < output_tensor_.size(); ++i) {
            if (output_tensor_[i].name == name) {
                return static_cast<int>(i);
            }
        }
        return -1;  // 返回 -1 表示未找到
    }

    void getCudaDeviceInfo();

  private:
    bool loadEngine(const std::string & engine_path);
    void setupInputOutputDims();

    // 逐算子提交：TRT 10.x 设 tensor 地址后 enqueueV3，TRT 8.x 用 buffers 数组 enqueueV2
    bool enqueue(cudaStream_t stream, void * input_data, const std::vector<void *> & output_data);

#ifdef HAS_CUDA_GRAPH
    // 把一次 enqueue 提交捕获成一张 CUDA Graph；失败返回 false 由调用方回退逐算子提交
    bool captureCudaGraph(cudaStream_t                stream,
                          void *                      input_data,
                          const std::vector<void *> & output_data);

    // 释放已捕获的图（先同步流，避免释放仍在执行的图）
    void destroyCudaGraph(cudaStream_t stream);

    // 当前调用的输入/输出地址是否与捕获时一致（地址漂移则图不可用）
    bool bindingsMatch(void * input_data, const std::vector<void *> & output_data) const;
#endif

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

    // ── CUDA Graph：把「推理提交」合成一次 cudaGraphLaunch（只用于异步 enqueue 路径）──
    bool use_cuda_graph_      = false;  // 配置开关（BackendInitOptions）
    int  graph_warmup_frames_ = 3;      // 捕获前预热帧数（下限 1）
    int  graph_warmup_count_  = 0;      // 已预热次数
    bool graph_disabled_ = false;  // 捕获/发射失败或地址漂移后永久回退逐算子提交
#ifdef HAS_CUDA_GRAPH
    cudaGraphExec_t           graph_exec_ = nullptr;  // 已捕获并实例化的可执行图
    std::vector<const void *> graph_bindings_;        // 捕获时绑定的输入 + 输出地址
#endif
};

}  // namespace inference
