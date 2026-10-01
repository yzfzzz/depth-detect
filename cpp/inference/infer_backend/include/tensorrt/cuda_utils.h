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

#endif  // HAS_CUDA
