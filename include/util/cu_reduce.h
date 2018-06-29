#pragma once

#include <cuda_runtime_api.h>
#include <math_functions.h>

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(~0, val, offset);
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {

    static __shared__ int shared[32]; // SFFared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__inline__ __device__
void _deviceReduceKernelStrideScaled(const float *in, float *out, int points, int stride, float scale) {
    float sum = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < points; i += blockDim.x * gridDim.x) {
        sum += scale*in[stride*i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}
__global__
void deviceReduceKernelStrideScaled(const float *in, float *out, int num_stride, int stride, float scale) {
    _deviceReduceKernelStrideScaled(in, out, num_stride, stride, scale);
}

__global__
void deviceReduceKernelStride(const float *in, float *out, int num_stride, int stride) {
    _deviceReduceKernelStrideScaled(in, out, num_stride, stride, 1.0f);
}

__global__
void deviceReduceKernelScaled(const float *in, float *out, size_t N, float scale) {
    _deviceReduceKernelStrideScaled(in, out, N, 1, scale);
}

__global__
void deviceReduceKernel(const float *in, float *out, size_t N) {
    _deviceReduceKernelStrideScaled(in, out, N, 1, 1.0f);
}

__inline__ __device__
void _deviceReduceKernelRepeatedLinearSum(const float *in, float *out, size_t N, int num_repeat, const float *coeffs) {
    float sum = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N*num_repeat; i += blockDim.x * gridDim.x) {
        sum += coeffs[i]*in[i%num_repeat];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

__global__
void deviceReduceKernelRepeatedLinearSum(const float *in, float *out, size_t N, int num_repeat, const float *coeffs) {
    _deviceReduceKernelRepeatedLinearSum(in, out, N, num_repeat, coeffs);
}

__global__
void deviceReduceKernelLinearSum(const float *in, float *out, size_t N, const float *coeffs) {
    _deviceReduceKernelRepeatedLinearSum(in, out, N, 1, coeffs);
}
