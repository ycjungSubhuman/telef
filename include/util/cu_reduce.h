#pragma once

#include <cuda_runtime_api.h>
#include <cstdio>
#include <assert.h>
#include <cstdlib>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include "cudautil.h"

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(~0, val, offset);
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {
    static __shared__ float shared[32]; // SFFared mem for 32 partial sums
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

/**
 * Calculate (in[0] + in[0+stride] + in[0+2*stride] + ... + in[0+(num_stride-1)*stride])*scale
 *
 * This does not calculate all sums in out[0].
 * If you specify gridDim larger than 1, partial sums are stored in 'out[blockIdx.x]'
 * So, if you want to add more than 1024 elements, use this more than twice
 * i.e)
 *      deviceReduceKernelStrideScaled<<<32, 1024>>>(in, out1, ...)
 *      deviceReduceKernelStrideScaled<<<1, 32>>>(out1, out2, ...)
 *
 * IMPORTANT: dimBlock should be multiple of warpSize, which was 32 when I wrote this code (CUDA 9.2)
 * i.e) deviceReduceKernelStrideScaled<<<1, 3>>>  <- WRONG
 *      deviceReduceKernelStrideScaled<<<1, 32>>> <- RIGHT
 *      deviceReduceKernelStrideScaled<<<1, 64>>> <- RIGHT
 *
 * @param in                input data array in device, with size num_stride*stride
 * @param out               output data array in device with size dimGrid
 * @param num_stride        number of strides you want to take
 * @param stride            size of a stride
 * @param scale             scale to be multiplied to the final sum
 */
__global__
void deviceReduceKernelStrideScaled(const float *in, float *out, int num_stride, int stride, float scale) {
    _deviceReduceKernelStrideScaled(in, out, num_stride, stride, scale);
}

/**
 * Wrapper for deviceReduceKernelStrideScaled with stride = 1, scale = 1.0f
 */
__global__
void deviceReduceKernel(const float *in, float *out, size_t N) {
    _deviceReduceKernelStrideScaled(in, out, N, 1, 1.0f);
}

__inline__ __device__
void _deviceReduceKernelLinearSum(const float *in, float *out, size_t N, int num_repeat, const float *coeffs) {
    float sum = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += coeffs[i]*in[i%N];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

/**
 * Calculate (in1[0]*in2[0] + in2[1]*in2[1] + ... + in1[N-1]*in2[N-1])
 *
 * This does not calculate all sums in out[0].
 * If you specify gridDim larger than 1, partial sums are stored in 'out[blockIdx.x]'
 * So, if you want to add more than 1024 elements, use this more than twice
 * i.e)
 *      deviceReduceKernelLinearSum<<<32, 1024>>>(in, out1, ...)
 *      deviceReduceKernelLinearSum<<<1, 32>>>(out1, out2, ...)
 *
 * IMPORTANT: dimBlock should be multiple of warpSize, which was 32 when I wrote this code (CUDA 9.2)
 * i.e) deviceReduceKernelLinearSum<<<1, 3>>>  <- WRONG
 *      deviceReduceKernelLinearSum<<<1, 32>>> <- RIGHT
 *      deviceReduceKernelLinearSum<<<1, 64>>> <- RIGHT
 *
 * @param in1               input data array in device, with size num_stride*stride
 * @param in2               input data array in device, with size num_stride*stride
 * @param out               output data array in device with size dimGrid
 * @param N                 number of elements in 'in1', or 'in2'
 */
__global__
void deviceReduceKernelLinearSum(const float *in1, const float *in2, float *out, size_t N) {
    float sum = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in1[i]*in2[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

/**
 * Calculate
 *  out[0] = (in[0]*coeff[0] + in[1]*coeff[1] + ... + in[N-1]*coeff[N-1])
 *  out[1] = (in[0]*coeff[N+0] + in[1]*coeff[N+1] + ... + in[N-1]*coeff[N+N-1])
 *  out[2] = (in[0]*coeff[2*N+0] + in[1]*coeff[2*N+1] + ... + in[N-1]*coeff[2*N+N-1])
 *  ...
 *  out[num_repeat-1] = (in[0]*coeff[(num_repeat-1)*N+0] + ... + in[N-1]*coeff[(num_repeat-1)*N+N-1])
 *
 * @param in                input data array in device
 * @param out               output data array in device
 * @param N                 number of elements in 'in'
 * @param num_repeat        number elements in 'out'
 * @param coeffs            coefficients for 'in'
 */
__host__
void repeatedLinearSum(const float *in, const float *coeffs, float *out, size_t N, int num_repeat) {
    const int blocks_needed = N / 1024 + 1;

    int threads_needed;
    if(blocks_needed > 1) {
        threads_needed = N % 1024 + 1;
    }
    else {
        threads_needed = 1024;
    }

    for(int i=0; i<num_repeat; i++) {
        const int dimBlock = ((threads_needed + 31) / 32) * 32;

        if (blocks_needed != 1) {
            const int dimGrid = ((blocks_needed + 31) / 32) * 32;
            float *temp = (float*)malloc(dimGrid*sizeof(float));
            deviceReduceKernelLinearSum <<< dimGrid, dimBlock >>> (in, coeffs + (N * i), temp, N);
            deviceReduceKernelLinearSum <<< 1, dimGrid >>> (in, coeffs + (N * i), out + i, N);
            CHECK_ERROR_MSG("Kernel Error");
        } else {
            const int dimGrid = blocks_needed;
            deviceReduceKernelLinearSum <<< dimGrid, dimBlock >>> (in, coeffs + (N * i), out + i, N);
            CHECK_ERROR_MSG("Kernel Error");
        }
    }
}

