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

__global__
void _calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    const int colDim = deformModel.dim;

    // grid-striding loop
    for (int i = start_index; i < deformModel.dim; i += stride) {

        position_d[i] = 0;
        for (int j = 0; j < deformModel.shapeRank; j++) {
            position_d[i] += params.fa1Params_d[j] * deformModel.shapeDeformBasis_d[i + colDim * j];
        }

        for (int j = 0; j < deformModel.expressionRank; j++) {
            position_d[i] += params.fa2Params_d[j] * deformModel.expressionDeformBasis_d[i + colDim * j];
        }

        position_d[i] +=
                deformModel.meanShapeDeformation_d[i]
                + deformModel.meanExpressionDeformation_d[i]
                + deformModel.ref_d[i];
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
        throw std::runtime_error("MatMul Failed\n");
    }
}

void applyRigidAlignment(float *align_pos_d, cublasHandle_t cnpHandle,
                         const float *position_d, const float *transMat, int N) {
    int size_homo = 4 * N;
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

    cudaFree(matB);
    cudaFree(matC);
}

void calculateLoss(float *residual, float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                   float *position_d, cublasHandle_t cnpHandle,
                   const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                   const bool isJacobianRequired) {

    float *residual_d, *fa1Jacobian_d, *fa2Jacobian_d, *ftJacobian_d, *fuJacobian_d;
    float *align_pos_d, *result_pos_d;
    float align_pos[deformModel.dim];

    /*
     * Allocate and Copy residual amd jacobian to GPU
     */
    CUDA_CHECK(cudaMalloc((void **) &residual_d, scanPointCloud.numLmks*3*sizeof(float)));

    // Compute Jacobians for each parameter
    CUDA_CHECK(cudaMalloc((void **) &fa1Jacobian_d, scanPointCloud.numLmks*3*params.numa1*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &fa2Jacobian_d, scanPointCloud.numLmks*3*params.numa2*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &ftJacobian_d, scanPointCloud.numLmks*3*params.numt*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &fuJacobian_d, scanPointCloud.numLmks*3*params.numu*sizeof(float)));

    // Allocate memory for Rigid aligned positions
    CUDA_CHECK(cudaMalloc((void **) &align_pos_d, deformModel.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &result_pos_d, deformModel.dim * sizeof(float)));
    // CuUDA Kernels run synchronously by default, to run asynchronously must explicitly specify streams

    /*
     * Compute Loss
     */
    // Calculate position_d
    calculateVertexPosition(position_d, params, deformModel);

    // Rigid alignment
    applyRigidAlignment(align_pos_d, cnpHandle, position_d, scanPointCloud.rigidTransform_d, deformModel.dim / 3);
    float r[9];
    float trans[16];
    float *trans_d;
    CUDA_CHECK(cudaMalloc((void **) &trans_d, 16*sizeof(float)));

    calc_r_from_u(r, params.fuParams_h);
    create_trans_from_tu(trans, params.ftParams_h, r);
    CUDA_CHECK(cudaMemcpy(trans_d, trans, 16* sizeof(float), cudaMemcpyHostToDevice));
    applyRigidAlignment(result_pos_d, cnpHandle, align_pos_d, trans_d, deformModel.dim / 3);

    // Calculate residual & jacobian for Landmarks
    calc_residual_lmk(residual_d, result_pos_d, scanPointCloud);
    if (isJacobianRequired) {
        calc_derivatives_lmk(ftJacobian_d, fuJacobian_d, fa1Jacobian_d, fa2Jacobian_d,
                             params.fuParams_d, align_pos_d, deformModel, scanPointCloud);
    }

    /*
     * Copy computed residual and jacobian to Host
     */
    CUDA_CHECK(cudaMemcpy(residual, residual_d, scanPointCloud.numLmks*3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fa1Jacobian, fa1Jacobian_d, scanPointCloud.numLmks*3*params.numa1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fa2Jacobian, fa2Jacobian_d, scanPointCloud.numLmks*3*params.numa2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ftJacobian, ftJacobian_d, scanPointCloud.numLmks*3*params.numt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fuJacobian, fuJacobian_d, scanPointCloud.numLmks*3*params.numu * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(align_pos, align_pos_d, deformModel.dim * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(residual_d));
    CUDA_CHECK(cudaFree(fa1Jacobian_d));
    CUDA_CHECK(cudaFree(fa2Jacobian_d));
    CUDA_CHECK(cudaFree(ftJacobian_d));
    CUDA_CHECK(cudaFree(fuJacobian_d));
    CUDA_CHECK(cudaFree(align_pos_d));
    CUDA_CHECK(cudaFree(result_pos_d));
}
