#pragma once

// CHECK_CUDA 及 CUDA 运行时头文件
// 仅 HAS_CUDA 构建参与编译

#ifdef HAS_CUDA

#    include <cuda_runtime_api.h>
#    include <spdlog/spdlog.h>

#    include <cstdio>

#    define CHECK_CUDA(call)                                                          \
        do {                                                                          \
            cudaError_t status = call;                                                \
            if (status != cudaSuccess) {                                              \
                auto logger = spdlog::get("app");                                     \
                if (logger) {                                                         \
                    logger->error("CUDA error at {}:{} - {}", __FILE__, __LINE__,     \
                                  cudaGetErrorString(status));                        \
                } else {                                                              \
                    fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                            cudaGetErrorString(status));                              \
                }                                                                     \
                exit(EXIT_FAILURE);                                                   \
            }                                                                         \
        } while (0)

// CUDA Graph需要 CUDA 10.0 起的 API：JetPack 4.x 的 CUDA 10.2 已满足，
// 更低版本编译时自动关闭该功能
#    if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
#        define HAS_CUDA_GRAPH 1
#    endif

#endif  // HAS_CUDA
