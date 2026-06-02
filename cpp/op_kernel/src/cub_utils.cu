#include "cub_utils.h"

#include <cub/cub.cuh>

void cub_get_min_max_temp_bytes(float *      d_in,
                                int          num_items,
                                size_t *     min_bytes,
                                size_t *     max_bytes,
                                cudaStream_t stream) {
    // initialize
    if (min_bytes) {
        *min_bytes = 0;
    }
    if (max_bytes) {
        *max_bytes = 0;
    }

    // Query temp size for Min (passing nullptr for output pointer to get required temp storage)
    size_t tmp_min = 0;
    cub::DeviceReduce::Min(nullptr, tmp_min, d_in, (float *) nullptr, num_items, stream);

    // Query temp size for Max
    size_t tmp_max = 0;
    cub::DeviceReduce::Max(nullptr, tmp_max, d_in, (float *) nullptr, num_items, stream);

    if (min_bytes) {
        *min_bytes = tmp_min;
    }
    if (max_bytes) {
        *max_bytes = tmp_max;
    }
}

void cub_device_reduce_min(void *       temp_storage,
                           size_t       temp_bytes,
                           float *      d_in,
                           float *      d_out,
                           int          num_items,
                           cudaStream_t stream) {
    cub::DeviceReduce::Min(temp_storage, temp_bytes, d_in, d_out, num_items, stream);
}

void cub_device_reduce_max(void *       temp_storage,
                           size_t       temp_bytes,
                           float *      d_in,
                           float *      d_out,
                           int          num_items,
                           cudaStream_t stream) {
    cub::DeviceReduce::Max(temp_storage, temp_bytes, d_in, d_out, num_items, stream);
}
