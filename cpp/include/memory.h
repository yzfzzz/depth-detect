#pragma once

#include "public.h"

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <memory>

// CUDA 显存删除器
struct CudaDeleter {
    void operator()(void * p) const noexcept {
        if (p != nullptr) {
            CHECK_CUDA(cudaFree(p));
        }
    }
};

// CUDA 显存删除器
struct PinnedCudaDeleter {
    void operator()(void * p) const noexcept {
        if (p != nullptr) {
            CHECK_CUDA(cudaFreeHost(p));
        }
    }
};

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

// 类型别名，简化声明
template <typename T> using unique_ptr_cuda        = std::unique_ptr<T, CudaDeleter>;
template <typename T> using unique_ptr_pinned_cuda = std::unique_ptr<T, PinnedCudaDeleter>;

using TrtEnginePtr  = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>;
using TrtRuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>;
using TrtContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>;
