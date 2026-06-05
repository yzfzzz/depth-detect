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
    BaseModel();
    virtual ~BaseModel();

    // 初始化模型
    virtual bool initInferenceBackend();

    // 获取后端类型
    BackendType getBackendType() const {
        return backend_ ? backend_->getBackendType() : BackendType::NONE;
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
    size_t getInputSize() const { return backend_ ? backend_->getInputSize() : 0; }

    // 获取输出数据大小（字节）
    size_t getOutputSize() const { return backend_ ? backend_->getOutputSize() : 0; }

    // 获取 CUDA 流
    cudaStream_t getStream() const { return stream_; }

    // 同步流
    void synchronizeStream() const {
        if (stream_ != 0) {
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
    }
  protected:
    // 创建后端（子类可重写以自定义后端选择逻辑）
    virtual std::unique_ptr<InferenceBackend> createBackend();

    // 检查 GPU 是否可用
    static bool isGPUAvailable();

    void waitAsync() {
        if (stream_ != 0) {
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
    }

    // 子类必须实现的部分
  public:
    virtual void init(const std::string & model_path, int raw_img_w, int raw_img_h);

    // 预处理路由（根据后端类型调用不同的预处理方法）
    virtual void preProcess(FrameInputContext & frame_input_context) {
        if (backend_->getBackendType() == BackendType::TENSORRT) {
            cudaPreProcess(frame_input_context);
        } else {
            cvMatPreProcess(frame_input_context);
        }
    }

    virtual void cvMatPreProcess(FrameInputContext & frame_input_context) {}

    virtual void cudaPreProcess(FrameInputContext & frame_input_context) = 0;  // cuda

    // 后处理路由（根据后端类型调用不同的后处理方法）
    virtual void postProcess(FrameInputContext & frame_input_context) {
        if (backend_->getBackendType() == BackendType::TENSORRT) {
            cudaPostProcess(frame_input_context);
        } else {
            cvMatPostProcess(frame_input_context);
        }
    }

    virtual void cvMatPostProcess(FrameInputContext & frame_input_context) {
        APP_ERROR(
            "CPU post-processing not implemented for base model, please implement in derived "
            "class.");
    };

    virtual void cudaPostProcess(FrameInputContext & frame_input_context) {
        APP_ERROR(
            "GPU post-processing not implemented for base model, please implement in derived "
            "class.");
    }  //gpu

    // 执行异步推理（供子类调用）
    virtual bool runInferenceAsync(FrameInputContext & frame_input_context) {
        APP_ERROR(
            "Asynchronous inference not implemented for base model, please implement in derived "
            "class.");
        return false;
    }

    // 执行同步推理（供子类调用）
    virtual bool runInference(FrameInputContext & frame_input_context) {
        APP_ERROR(
            "Synchronous inference not implemented for base model, please implement in derived "
            "class.");
        return false;
    }

    virtual void getInferOutputResult(InferOutputContext & infer_output_context) {
        APP_ERROR(
            "getInferOutput not implemented for base model, please implement in derived class if "
            "using asynchronous inference.");
    }

  protected:
    // 原始图像分辨率
    int raw_img_w_;
    int raw_img_h_;

    // 模型输入分辨率
    int         input_h_;
    int         input_w_;
    std::string model_path_;
    bool        initialized_ = false;

    std::unique_ptr<InferenceBackend> backend_;
    cudaStream_t                      stream_;  // 由 BaseModel 管理
};
