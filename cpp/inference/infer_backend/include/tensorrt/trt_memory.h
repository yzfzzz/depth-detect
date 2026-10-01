#pragma once

#include <NvInfer.h>

#include <memory>

// TRT 对象删除器
struct TrtDeleter {
    template <typename T> void operator()(T * p) const noexcept {
        if (p == nullptr) {
            return;
        }
#if NV_TENSORRT_MAJOR >= 10
        delete p;  // TRT 10.x 使用标准 C++ 析构
#else
        p->destroy();  // TRT 8.x 使用 destroy 方法
#endif
    }
};

// TensorRT 专属智能指针：仅 tensorrt_backend 使用，不随 memory.h 扩散到模型层
using TrtEnginePtr  = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>;
using TrtRuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>;
using TrtContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>;
