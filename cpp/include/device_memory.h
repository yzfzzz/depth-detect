#pragma once

#include <memory>

#ifdef HAS_CUDA
#    include "tensorrt/cuda_utils.h"

#    include <cuda_runtime.h>

// CUDA 显存删除器
struct DeviceDeleter {
    void operator()(void * p) const noexcept {
        if (p != nullptr) {
            CHECK_CUDA(cudaFree(p));
        }
    }
};

// CUDA 显存删除器
struct HostPinnedDeleter {
    void operator()(void * p) const noexcept {
        if (p != nullptr) {
            CHECK_CUDA(cudaFreeHost(p));
        }
    }
};
#else
#    include <cstddef>
#    include <cstdlib>

// 无 CUDA 环境的回退删除器
struct DeviceDeleter {
    void operator()(void * p) const noexcept { std::free(p); }
};

struct HostPinnedDeleter {
    void operator()(void * p) const noexcept { std::free(p); }
};

// CUDA vector_types.h 的 uchar3 回退：模型层的伪彩缓冲区声明仅以
struct uchar3 {
    unsigned char x, y, z;
};
#endif  // HAS_CUDA

// 类型别名，简化声明
template <typename T> using unique_ptr_device      = std::unique_ptr<T, DeviceDeleter>;
template <typename T> using unique_ptr_pinned_host = std::unique_ptr<T, HostPinnedDeleter>;

#ifdef HAS_CUDA

// 主机端 pinned memory：作为 D2H 异步拷贝的目标，避免同步等待
// 封装 cudaMallocHost，分配页锁定内存用于异步 D2H 拷贝，避免 cudaMemcpy 阻塞 CPU
inline void * allocPinnedHost(size_t bytes) {
    void * ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&ptr, bytes));
    return ptr;
}

// 封装 cudaMalloc 为返回裸指针的 lambda，配合 unique_ptr_device 自动管理显存生命周期
inline void * allocDevice(size_t bytes) {
    void * ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, bytes));
    return ptr;
}

#else

// 无 CUDA 环境的回退分配：退化为普通主机内存，仅保证 CPU-only 构建可链接
inline void * allocPinnedHost(size_t bytes) {
    return std::malloc(bytes);
}

inline void * allocDevice(size_t bytes) {
    return std::malloc(bytes);
}

#endif  // HAS_CUDA
