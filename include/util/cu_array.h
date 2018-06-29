#pragma once

#include <cuda_runtime_api.h>

template<typename T>
__global__
void scale_array(T *array_d, int size, float scale) {
    const int start = blockDim.x * blockIdx.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    for (int i=start; i<size; i+=step) {
        array_d[i] *= scale;
    }
}