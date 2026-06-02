#ifndef CUB_UTILS_H
#define CUB_UTILS_H

#include <cuda_runtime.h>

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Query temporary storage size (in bytes) required by CUB DeviceReduce Min/Max
// d_in: pointer to device input array (float*)
// num_items: number of elements in input
// min_bytes, max_bytes: outputs for required workspace sizes
// stream: CUDA stream to associate (can be 0)
void cub_get_min_max_temp_bytes(float *      d_in,
                                int          num_items,
                                size_t *     min_bytes,
                                size_t *     max_bytes,
                                cudaStream_t stream);

// Perform DeviceReduce Min/Max using provided temporary storage
// temp_storage: device pointer to workspace (size at least temp_bytes)
// temp_bytes: size of workspace in bytes
// d_in: input device pointer
// d_out: output device pointer (single float)
void cub_device_reduce_min(void *       temp_storage,
                           size_t       temp_bytes,
                           float *      d_in,
                           float *      d_out,
                           int          num_items,
                           cudaStream_t stream);

void cub_device_reduce_max(void *       temp_storage,
                           size_t       temp_bytes,
                           float *      d_in,
                           float *      d_out,
                           int          num_items,
                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // CUB_UTILS_H
