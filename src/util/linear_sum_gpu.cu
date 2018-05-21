#include "util/linear_sum_gpu.h"

__global__ void linearSum(const float * const basisMatrix, int basisColSize, const float * const coeff, int coeffLength, float *result) {
    int vtxIdx = (blockIdx.x * blockDim.x) + (threadIdx.x);
    int coeffIdx = (blockIdx.y * blockDim.y) + (threadIdx.y);

    if (vtxIdx < basisColSize && coeffIdx <coeffLength) {
        result[vtxIdx] += coeff[coeffIdx] * basisMatrix[coeffLength * vtxIdx + coeffIdx];
    }
}

void getLinearSum (const float *const basisMatrix, int basisColSize, const float *const coeff, int coeffSize, float *result)
{
    // Alloc device memory
    float *result_d;
    float *basis_d;
    float *coeff_d;
    const int resultMemSize = static_cast<int>(basisColSize*sizeof(float));
    const int basisMemSize = static_cast<int>(basisColSize*coeffSize*sizeof(float));
    const int coeffMemSize = coeffSize*sizeof(float);
    cudaMalloc((void**)(&result_d), resultMemSize);
    cudaMalloc((void**)(&basis_d), basisMemSize);
    cudaMalloc((void**)(&coeff_d), coeffMemSize);

    // Copy to device memory
    cudaMemcpy(basis_d, basisMatrix, basisMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(coeff_d, coeff, coeffSize, cudaMemcpyHostToDevice);

    // Launch kernel
    const int M = basisColSize;
    const int N = coeffSize;
    const int blockSizeM = 16;
    const int blockSizeN = 16;
    dim3 dimBlock(blockSizeM, blockSizeN);
    dim3 dimGrid((M + blockSizeM - 1) / blockSizeM, (N + blockSizeN - 1) / blockSizeN);
    linearSum<<<dimGrid, dimBlock>>>(basis_d, M, coeff_d, N, result_d);

    cudaMemcpy(result, result_d, resultMemSize, cudaMemcpyDeviceToHost);
}
