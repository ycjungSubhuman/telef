#include "face/cu_model_kernel.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "util/cudautil.h"
#include "util/cu_quaternion.h"
#include "align/cu_loss.h"
#include "util/transform.h"

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

    static __shared__ float shared[32]; // SFFared mem for 32 partial sums
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

        position_d[i] = 0;
        for (int j = 0; j < deformModel.rank; j++) {
            position_d[i] += params.faParams_d[j] * deformModel.deformBasis_d[i + colDim * j];
        }

        position_d[i] += deformModel.mean_d[i] + deformModel.ref_d[i];
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

        if (isJacobianRequired) {
            for (int j = 0; j < deformB_row; j++) {
                float basis[3] = {deformBasis_d[colDim * j + 3 * posIdx + 0],  // x @ col j
                                  deformBasis_d[colDim * j + 3 * posIdx + 1],  // y @ col j
                                  deformBasis_d[colDim * j + 3 * posIdx + 2]}; // z @ col j

                // Element wise multiplication and sum
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += ptSubt[k] * basis[k];
                }

                float jacobi = -2 * landmarkCoeff * sum;

                // Reduce Jacobians across across block
                jacobi = blockReduceSum(jacobi);

                // Add partial sum into atomic output, only do it once per block
                if (threadIdx.x == 0) {
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
    dim3 lmkThrds(64);
    dim3 lmkBlocks((idim + 64 - 1) / 64);

    _calculateLandmarkLoss << < lmkBlocks, lmkThrds >> > (residual_d, jacobian_d, position_d,
                           deformModel.deformBasis_d, deformModel.rank, deformModel.dim, scanPointCloud.validModelLmks_d,
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

void cudaMatMul(float *matC, cublasHandle_t cnpHandle,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols) {

    // Don't know what this is (scalar?) but examples use this
    cublasStatus_t status;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

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
        return;
    }
}

void applyRigidAlignment(float *align_pos_d, cublasHandle_t cnpHandle,
                         const float *position_d, const float *transMat, int N) {
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
    cudaMatMul(matC, cnpHandle, transMat, 4, 4, matB, 4, N);

// hnormalized point (x,y,z)
    _hnormalizedPositions << < grid, block >> > (align_pos_d, matC, N);
    CHECK_ERROR_MSG("Kernel Error");

//    printf("cublasSgemm Status %i\n", status);

    cudaFree(matB);
    cudaFree(matC);
}

void calculateLoss(float *residual, float *faJacobian, float *ftJacobian, float *fuJacobian, float *position_d,
                   cublasHandle_t cnpHandle,
                   const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                   const bool isJacobianRequired) {

//    std::cout << "calculateLoss" << std::endl;
    float *residual_d, *faJacobian_d, *ftJacobian_d, *fuJacobian_d;
    float *align_pos_d, *result_pos_d;
    float align_pos[deformModel.dim];

    /*
     * Allocate and Copy residual amd jacobian to GPU
     */
    // Currently, We are using only 1 residual to contain the loss
    CUDA_CHECK(cudaMalloc((void **) &residual_d, sizeof(float)));

    // Compute Jacobians for each parameter
    CUDA_CHECK(cudaMalloc((void **) &faJacobian_d, params.numa * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &ftJacobian_d, params.numt * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &fuJacobian_d, params.numu * sizeof(float)));

    // Allocate memory for Rigid aligned positions
    CUDA_CHECK(cudaMalloc((void **) &align_pos_d, deformModel.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &result_pos_d, deformModel.dim * sizeof(float)));
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
    applyRigidAlignment(align_pos_d, cnpHandle, position_d, scanPointCloud.rigidTransform_d, deformModel.dim / 3);
    float r[9];
    float trans[16];
    float *trans_d;
    CUDA_CHECK(cudaMalloc((void **) &trans_d, 16*sizeof(float)));

    calc_r_from_u(r, params.fuParams_h);
    create_trans_from_tu(trans, params.ftParams_h, r);
    CUDA_CHECK(cudaMemcpy(trans_d, trans, 16* sizeof(float), cudaMemcpyHostToDevice));

    applyRigidAlignment(result_pos_d, cnpHandle, align_pos_d, trans_d, deformModel.dim / 3);
    //cudaDeviceSynchronize();

    // Calculate residual_d, jacobian_d for Landmarks
    calc_mse_lmk(residual_d, result_pos_d, scanPointCloud);
    CHECK_ERROR_MSG("Kernel Error");

    if (isJacobianRequired) {
        calc_derivatives_lmk(ftJacobian_d, fuJacobian_d, faJacobian_d,
                             params.fuParams_d, align_pos_d, result_pos_d, deformModel, scanPointCloud);
        CHECK_ERROR_MSG("Kernel Error");
    }

//    std::cout << "calculateLoss: calculateLandmarkLoss (Jacobi:"<<isJacobianRequired<<")" << std::endl;
    //calculateLandmarkLoss(residual_d, faJacobian_d, align_pos_d, deformModel, scanPointCloud, isJacobianRequired);
    //cudaDeviceSynchronize();

    /*
     * Copy computed residual and jacobian to Host
     */
//    std::cout << "Copy to host" << std::endl;
    CUDA_CHECK(cudaMemcpy(residual, residual_d, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(faJacobian, faJacobian_d, params.numa * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ftJacobian, ftJacobian_d, params.numt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fuJacobian, fuJacobian_d, params.numu * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(align_pos, align_pos_d, deformModel.dim * sizeof(float), cudaMemcpyDeviceToHost));

//    std::cout << "align_pos: " << align_pos[15] << std::endl;
    //delete align_pos;
    //align_pos = NULL;
    cudaFree(residual_d);
    cudaFree(faJacobian_d);
    cudaFree(ftJacobian_d);
    cudaFree(fuJacobian_d);
    cudaFree(align_pos_d);
    cudaFree(result_pos_d);
}
