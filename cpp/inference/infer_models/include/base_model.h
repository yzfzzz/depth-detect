#pragma once

#include "frame.h"
#include "inference_backend.h"
#include "logger_manager.h"
#include "memory.h"
#include "public.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 推理模型基类，提供统一的模型接口，支持多后端（TensorRT/ONNX Runtime），自动检测 GPU 可用性并选择合适的后端
class BaseModel {
  public:
    virtual ~BaseModel();

    // 初始化模型
    virtual bool initInferenceBackend(std::map<std::string, std::string> model_path, bool use_gpu);

    // 获取后端类型
    BackendType getBackendType() const {
        return backend_ ? backend_->getBackendType() : BackendType::Unkown;
    }

    // 检查模型是否已初始化
    bool isBackendInitialized() const { return initialized_; }

    // 获取输入维度
    std::vector<int> getInputDims() const {
        return backend_ ? backend_->getInputDims() : std::vector<int>{};
    }

    // 获取输出维度
    std::vector<int> getOutputDims() const {
        return backend_ ? backend_->getOutputDims() : std::vector<int>{};
    }

    // 获取输入数据大小（字节）
    size_t getInputByteSize() const { return backend_ ? backend_->getInputByteSize() : 0; }

    // 获取输出数据大小（字节）
    size_t getOutputByteSize() const { return backend_ ? backend_->getOutputByteSize() : 0; }

    // 获取 CUDA 流
    cudaStream_t getStream() const { return stream_; }

  protected:
    // 创建后端（子类可重写以自定义后端选择逻辑）
    virtual std::unique_ptr<InferenceBackend> createBackend(
        std::map<std::string, std::string> model_path,
        bool                               use_gpu);

    // 检查 GPU 是否可用
    static bool isGPUAvailable();

    // 同步流
    void synchronizeStream() const {
        if (stream_ != 0) {
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
    }

    // 子类必须实现的部分
  public:
    void init(std::map<std::string, std::string> model_path,
              int                                raw_img_w,
              int                                raw_img_h,
              bool                               use_gpu = false);

    // 预处理路由（根据后端类型调用不同的预处理方法）
    // virtual void preProcess(FrameInputContext & frame_input_context) {
    //     if (backend_->getBackendType() == BackendType::TENSORRT) {
    //         cudaPreProcess(frame_input_context);
    //     } else {
    //         cvMatPreProcess(frame_input_context);
    //     }
    // }

    virtual std::vector<float> cvMatPreProcess(FrameInputContext & frame_input_context) = 0;

    virtual void cudaPreProcess(FrameInputContext & frame_input_context) = 0;  // cuda

    // 后处理路由（根据后端类型调用不同的后处理方法）
    // virtual void postProcess(FrameInputContext & frame_input_context) {
    //     if (backend_->getBackendType() == BackendType::TENSORRT) {
    //         cudaPostProcess(frame_input_context);
    //     } else {
    //         cvMatPostProcess(frame_input_context);
    //     }
    // }

    virtual void cvMatPostProcess(InferOutputContext & infer_output_context) = 0;  //cpu

    virtual void cudaPostProcess(FrameInputContext & frame_input_context) = 0;     //gpu

    // 执行异步推理（供子类调用）
    virtual bool runInferenceAsync(FrameInputContext & frame_input_context);

    // 执行同步推理（供子类调用）
    virtual bool runInference(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context);

    virtual void getInferOutputResult(InferOutputContext & infer_output_context) = 0;

  protected:
    // 原始图像分辨率
    int raw_img_w_;
    int raw_img_h_;

    // 模型输入分辨率
    int                                  input_h_;
    int                                  input_w_;
    bool                                 initialized_ = false;
    // 模型输入输出缓冲区: d_infer_io_[0] -> input, d_infer_io_[1] -> output
    std::array<unique_ptr_cuda<void>, 2> d_infer_io_;
    std::unique_ptr<InferenceBackend>    backend_;
    cudaStream_t                         stream_;
    std::vector<float>                   h_infer_out_;
};
