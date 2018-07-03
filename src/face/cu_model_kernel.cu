#include "face/cu_model_kernel.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "util/cudautil.h"

#define BLOCKSIZE 128

static std::string _cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {

    static __shared__ int shared[32]; // SFFared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__
void _calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    const int colDim = deformModel.dim;

    // grid-striding loop
    for (int i = start_index; i < deformModel.dim; i += stride) {

        //printf("_calculateVertexPosition %d\n", i);
        position_d[i] = 0;;
        for (int j = 0; j < deformModel.rank; j++) {
            position_d[i] += params.params_d[j] * deformModel.deformBasis_d[i + colDim * j];
        }
    }
}

void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel) {
    int idim = deformModel.dim;
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((idim + BLOCKSIZE - 1) / BLOCKSIZE);
    _calculateVertexPosition << < dimGrid, dimBlock >> > (position_d, params, deformModel);
    CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _calculateLandmarkLoss(float *residual_d, float *jacobian_d, const float *position_d,
                            const float *deformBasis_d, int deformB_row, int deformB_col,
                            const int *lmks_d, const float *scanPoints_d, const int *scanLmks_d,
                            int numLmks, const bool isJacobianRequired) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid


    //TODO: Add param for weights
    float landmarkCoeff = 1.0;
    const int colDim = deformB_col;

    // This Treads Placeholders for output
    float res = 0.0;

    // grid-striding loop
    //printf("NumLmks: %d\n", nPoints);
    for (int idx = start_index; idx < numLmks; idx += stride) {

        //printf("Index: %d\n", idx);
        int posIdx = lmks_d[idx];
        int scanIdx = scanLmks_d[idx];

        float ptSubt[3] = {position_d[3 * posIdx] - scanPoints_d[3 * scanIdx],           // x
                           position_d[3 * posIdx + 1] - scanPoints_d[3 * scanIdx + 1],   // y
                           position_d[3 * posIdx + 2] - scanPoints_d[3 * scanIdx + 2]};  // z

        float squaredNorm = 0.0;

        for (int i = 0; i < 3; i++) {
            squaredNorm += ptSubt[i] * ptSubt[i];
        }

        res += landmarkCoeff * squaredNorm;

//        printf("res[%d]: %.6f\n", posIdx, res);

        if (isJacobianRequired) { ;
            for (int j = 0; j < deformB_row; j++) {
                float basis[3] = {deformBasis_d[colDim * j + 3 * posIdx + 0],  // x @ col j
                                  deformBasis_d[colDim * j + 3 * posIdx + 1],  // y @ col j
                                  deformBasis_d[colDim * j + 3 * posIdx + 2]}; // z @ col j

                // Element wise multiplication and sum
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += ptSubt[k] * basis[k];
                }
//                printf("basis[%d]: %.6f\n", colDim * j + 3 * posIdx + 0, deformBasis_d[colDim * j + 3 * posIdx + 0]);
//                printf("basis[%d]: %.6f\n", colDim * j + 3 * posIdx + 1, deformBasis_d[colDim * j + 3 * posIdx + 1]);
//                printf("basis[%d]: %.6f\n", colDim * j + 3 * posIdx + 2, deformBasis_d[colDim * j + 3 * posIdx + 2]);
//
//                printf("sum[%d]: %.6f\n", j, sum);

                float jacobi = -2 * landmarkCoeff * sum;

//                printf("jacobi[%d]: %.6f\n", j, jacobi);

                // Reduce Jacobians across across block
                jacobi = blockReduceSum(jacobi);

                // Add partial sum into atomic output, only do it once per block
                if (threadIdx.x == 0) {
//                    printf("Reduced jacobi[%d]: %.6f\n", j, jacobi);
                    atomicAdd(&jacobian_d[j], jacobi/numLmks);
                }
            }
        }
    }

    // Reduce Residuals across block
    res = blockReduceSum(res);

    // Add partial sum into atomic output, only do it once per block
    if (threadIdx.x == 0) {
//        printf("Reduced res[%d]: %.6f\n", start_index, res/numLmks);
        atomicAdd(residual_d, res/numLmks);
    }
}

void calculateLandmarkLoss(float *residual_d, float *jacobian_d, const float *position_d,
                           const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                           const bool isJacobianRequired) {

    int idim = scanPointCloud.numLmks;
    dim3 lmkThrds(BLOCKSIZE);
    dim3 lmkBlocks((idim + BLOCKSIZE - 1) / BLOCKSIZE);

    _calculateLandmarkLoss << < lmkBlocks, lmkThrds >> > (residual_d, jacobian_d, position_d,
                           deformModel.deformBasis_d, deformModel.rank, deformModel.dim, deformModel.lmks_d,
                           scanPointCloud.scanPoints_d, scanPointCloud.scanLmks_d, scanPointCloud.numLmks,
                           isJacobianRequired);
    CHECK_ERROR_MSG("Kernel Error");
}


__global__
void _homogeneousPositions(float *h_position_d, const float *position_d, int nPoints) {

    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int index = start_index; index < nPoints; index += stride) {
        // homogeneous coordinates (x,y,z,1);
        float pos[4] = {position_d[3 * index], position_d[3 * index + 1], position_d[3 * index + 2], 1};
        memcpy(&h_position_d[4 * index], &pos[0], 4 * sizeof(float));
    }
}

__global__
void _hnormalizedPositions(float *position_d, const float *h_position_d, int nPoints) {

    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int index = start_index; index < nPoints; index += stride) {

        // homogeneous coordinates (x,y,z,1);
        float hnorm = h_position_d[4 * index + 3];
        position_d[3 * index] = h_position_d[4 * index] / hnorm;
        position_d[3 * index + 1] = h_position_d[4 * index + 1] / hnorm;
        position_d[3 * index + 2] = h_position_d[4 * index + 2] / hnorm;
    }
}

void cudaMatMul(float *matC,
                const float *matA_host, int aRows, int aCols,
                const float *matB, int bRows, int bCols) {

    cublasHandle_t cnpHandle;
    cublasStatus_t status = cublasCreate(&cnpHandle);

    // Don't know what this is (scalar?) but examples use this
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    float *matA;

    cudaMalloc((void **) &matA, aCols * aRows * sizeof(float));

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed, %s\n", _cublasGetErrorEnum(status).c_str());
        return;
    }

    // Copy to GPU
    status = cublasSetMatrix(aRows, aCols, sizeof(float), matA_host, /*ldim*/ aCols, matA, /*ldim*/ aCols);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed: %s", _cublasGetErrorEnum(status).c_str());
        cudaFree(matA);
        cublasDestroy(cnpHandle);
        return;
    }

    /* Perform operation using cublas, inputs/outputs are col-major.
     * vector and array were originally Eigen which defaults to Col-major
     * m is rows for A and C
     * n is cols for B and C
     * k is cols for A and rows for B*/
    // Matrix Mult C = α op ( A ) op ( B ) + β C
    status =
            cublasSgemm(cnpHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N, // Matrix op(A) and op(B): No-op, Transpose, Conjugate
                        aRows, bCols, aCols, //(m,n,k)
                        alpha,
                        matA, aRows/*leading dim, ROWS?*/, //(4x4) or (mxk)
                        matB, bRows/*leading dim*/, //(4xN) or (kxn)
                        beta,
                        matC, bRows/*leading dim*/); //(4xN) or (mxk)

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("MatMul Failed, %s\n", _cublasGetErrorEnum(status).c_str());
        cublasDestroy(cnpHandle);
        return;
    }

    cublasDestroy(cnpHandle);
}

void applyRigidAlignment(float *align_pos_d, const float *position_d, const float *transMatA, int N) {
    int size_homo = 4 * N;
    int size = 3 * N;
    dim3 grid = ((N + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 block = BLOCKSIZE;

    float *matB, *matC;

    cudaMalloc((void **) &matB, size_homo * sizeof(float));
    cudaMalloc((void **) &matC, size_homo * sizeof(float));


    // Create homogenous matrix (x,y,z,1)
    _homogeneousPositions << < grid, block >> > (matB, position_d, N);
    CHECK_ERROR_MSG("Kernel Error");

    /* Perform operation using cublas, inputs/outputs are col-major.
     * vector and array were originally Eigen which defaults to Col-major
     * m is rows for A and C
     * n is cols for B and C
     * k is cols for A and rows for B*/
    // Matrix Mult C = α op ( A ) op ( B ) + β C
    cudaMatMul(matC, transMatA, 4, 4, matB, 4, N);

// hnormalized point (x,y,z)
    _hnormalizedPositions << < grid, block >> > (align_pos_d, matC, N);
    CHECK_ERROR_MSG("Kernel Error");

//    printf("cublasSgemm Status %i\n", status);

    cudaFree(matB);
    cudaFree(matC);
}

void calculateLoss(float *residual, float *jacobian, float *position_d,
                   const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                   const bool isJacobianRequired) {

//    std::cout << "calculateLoss" << std::endl;
    float *residual_d, *jacobian_d;
    float *align_pos_d;
    float align_pos[deformModel.dim];

    /*
     * Allocate and Copy residual amd jacobian to GPU
     */
    // Currently, We are using only 1 residual to contain the loss
    CUDA_CHECK(cudaMalloc((void **) &residual_d, sizeof(float)));

    // Compute Jacobians for each parameter
    CUDA_CHECK(cudaMalloc((void **) &jacobian_d, params.numParams * sizeof(float)));

    // Allocate memory for Rigid aligned positions
    CUDA_CHECK(cudaMalloc((void **) &align_pos_d, deformModel.dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(residual_d, residual, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(jacobian_d, jacobian, params.numParams * sizeof(float), cudaMemcpyHostToDevice));

    // CuUDA Kernels run synchronously by default, to run asynchronously must explicitly specify streams

    /*
     * Compute Loss
     */
    // Calculate position_d
//    std::cout << "calculateLoss: calculateVertexPosition" << std::endl;
    calculateVertexPosition(position_d, params, deformModel);
    //cudaDeviceSynchronize();

    // Rigid alignment
//    std::cout << "calculateLoss: applyRigidAlignment" << std::endl;
    applyRigidAlignment(align_pos_d, position_d, scanPointCloud.rigidTransform_d, deformModel.dim / 3.0);
    //cudaDeviceSynchronize();

    // Calculate residual_d, jacobian_d for Landmarks
//    std::cout << "calculateLoss: calculateLandmarkLoss (Jacobi:"<<isJacobianRequired<<")" << std::endl;
    calculateLandmarkLoss(residual_d, jacobian_d, align_pos_d, deformModel, scanPointCloud, isJacobianRequired);
    //cudaDeviceSynchronize();

    /*
     * Copy computed residual and jacobian to Host
     */
//    std::cout << "Copy to host" << std::endl;
    CUDA_CHECK(cudaMemcpy(residual, residual_d, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(jacobian, jacobian_d, params.numParams * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(align_pos, align_pos_d, deformModel.dim * sizeof(float), cudaMemcpyDeviceToHost));



    //TODO: return value to see rigid aligned mesh?

//    std::cout << "align_pos: " << align_pos[15] << std::endl;
    //delete align_pos;
    //align_pos = NULL;

    cudaFree(align_pos_d);
}
