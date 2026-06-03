#pragma once

#include "inference_backend.h"
#include "logger_manager.h"
#include "memory.h"
#include "public.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief 推理模型基类
 * 
 * 提供统一的模型接口，支持多后端（TensorRT/ONNX Runtime）
 * 自动检测 GPU 可用性并选择合适的后端
 */
class BaseModel {
  public:
    /**
     * @brief 构造函数
     */
    BaseModel();
    virtual ~BaseModel();

    /**
     * @brief 初始化模型
     * @return 是否初始化成功
     */
    virtual bool initInferenceBackend();

    /**
     * @brief 获取后端类型
     */
    BackendType getBackendType() const {
        return backend_ ? backend_->getBackendType() : BackendType::NONE;
    }

    /**
     * @brief 检查模型是否已初始化
     */
    bool isBackendInitialized() const { return initialized_; }

    /**
     * @brief 获取输入维度
     */
    std::vector<int> getInputDims() const {
        return backend_ ? backend_->getInputDims() : std::vector<int>{};
    }

    /**
     * @brief 获取输出维度
     */
    std::vector<int> getOutputDims() const {
        return backend_ ? backend_->getOutputDims() : std::vector<int>{};
    }

    /**
     * @brief 获取输入数据大小（字节）
     */
    size_t getInputSize() const { return backend_ ? backend_->getInputSize() : 0; }

    /**
     * @brief 获取输出数据大小（字节）
     */
    size_t getOutputSize() const { return backend_ ? backend_->getOutputSize() : 0; }

    /**
     * @brief 获取 CUDA 流
     */
    cudaStream_t getStream() const { return stream_; }

    /**
     * @brief 同步流
     */
    void synchronizeStream() const {
        if (stream_ != 0) {
            CHECK_CUDA(cudaStreamSynchronize(stream_));
        }
    }
  protected:
    /**
     * @brief 创建后端（子类可重写以自定义后端选择逻辑）
     */
    virtual std::unique_ptr<InferenceBackend> createBackend();

    /**
     * @brief 检查 GPU 是否可用
     */
    static bool isGPUAvailable();

    // 子类必须实现的部分
  public:
    virtual void init(const std::string & model_path, int raw_img_w, int raw_img_h);
    /**
     * @brief 预处理（子类必须实现）
     */
    virtual void preProcess(const cv::Mat & input, void * output) = 0;  // cpu
    virtual void cudaPreProcess(uchar * input)                    = 0;  // cuda

    /**
     * @brief 后处理（子类必须实现）
     */
    virtual void postProcess(void * output, void * results) = 0;  //cpu
    virtual void cudaPostProcess()                          = 0;  //gpu

    /**
     * @brief 执行异步推理（供子类调用）
     */
    bool runInferenceAsync(void * input_data, void * output_data) {
        if (!backend_) {
            APP_ERROR("Backend not initialized");
            return false;
        }
        return backend_->runInferenceAsync(input_data, output_data, stream_);
    }

    /**
     * @brief 执行同步推理（供子类调用）
     */
    bool runInference(void * input_data, void * output_data) {
        if (!backend_) {
            APP_ERROR("Backend not initialized");
            return false;
        }
        return backend_->runInference(input_data, output_data);
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
