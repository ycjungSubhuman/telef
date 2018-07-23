#include "face/cu_model_kernel.h"

#include <iostream>
#include <cmath>
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

__global__
static void calculateLandmarkIndices(int *mesh_inds, int *scan_inds, C_PcaDeformModel model, C_ScanPointCloud scan) {
    const int start = blockDim.x * blockIdx.x + threadIdx.x;
    const int size = scan.numLmks;
    const int step = blockDim.x * gridDim.x;

    for(int ind=start; ind<size; ind+=step) {
        mesh_inds[ind] = model.lmks_d[scan.modelLandmarkSelection_d[ind]];
        scan_inds[ind] = scan.modelLandmarkSelection_d[ind];
    }
}

/**
 * Project xyz coord into uv space
 * @param uv
 * @param xyz
 * @param fx
 * @param fy
 * @param cx
 * @param cy
 */
__device__
void convertXyzToUv(int *uv, const float* xyz, float fx, float fy, float cx, float cy) {
    uv[0] = static_cast<int>(std::round(xyz[0] * fx / xyz[2] + cx));
    uv[1] = static_cast<int>(std::round(xyz[1] * fy / xyz[2] + cy));
}

__global__
void _find_mesh_to_scan_corr(int *meshCorr_d, int *scanCorr_d, float *distance_d, int *numCorr,
                             const float *position_d, int num_points, C_ScanPointCloud scan, float radius, int maxPoints) {
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = num_points;
    const int step = blockDim.x * gridDim.x;
    // Initialize numCorr to 0, will use atomicAdd to increment counter
    if(threadIdx.x == 0) numCorr[0] = 0;
    __syncthreads();

    for(int i=start; i<size; i+=step) {
        // Project Point into UV space for mapping finding closest xy on scan
        int uv[2];

        convertXyzToUv(&uv[0], &position_d[i*3], scan.fx, scan.fy, scan.cx, scan.cy);

        // Calculate 1-D Index of scan point from UV coord
        int scanIndx = uv[1] * scan.width * 3 + uv[0] * 3 + 0;
        int scanIndy = uv[1] * scan.width * 3 + uv[0] * 3 + 1;
        int scanIndz = uv[1] * scan.width * 3 + uv[0] * 3 + 2;

        // Check if Model coord outside of UV space, could happen if model is aligned to face near image boarder
        if(scanIndx < scan.numPoints*3 && scanIndy < scan.numPoints*3 && scanIndz < scan.numPoints*3 &&
           scanIndx >=0 && scanIndy >=0 && scanIndz >=0) {
            // Check for NaN Points
            bool isNaN = std::isfinite(scan.scanPoints_d[scanIndx]) == 0
                         || std::isfinite(scan.scanPoints_d[scanIndy]) == 0
                         || std::isfinite(scan.scanPoints_d[scanIndz]) == 0;

            // Add correspondance if within search radius, if radius is 0, include all points
            if (!isNaN) {
                // Check z distance for within radius tolerance (Use xyz EuclidDist instead?)
                float dist = std::fabs(position_d[i * 3 + 2] - scan.scanPoints_d[scanIndx + 2]);
//                printf("Correspondance %.4f\n", dist);
                if (radius <= 0 || dist <= radius) {
                    int idx = atomicAdd(&numCorr[0], 1);
//                    printf("Correspondance %d\n", idx);
                    if (maxPoints <= 0 || idx < maxPoints) {
                        meshCorr_d[idx] = i;
                        scanCorr_d[idx] = scanIndx / 3;
                        distance_d[idx] = dist;
                    }
                }
            }
        }
    }
}

void find_mesh_to_scan_corr(int *meshCorr_d, int *scanCorr_d, float *distance_d, int *numCorr,
                           const float *position_d, int num_points, C_ScanPointCloud scan, float radius, int maxPoints) {
    int idim = num_points/3;
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((idim + BLOCKSIZE - 1) / BLOCKSIZE);

    _find_mesh_to_scan_corr << < dimGrid, dimBlock >> > (meshCorr_d, scanCorr_d, distance_d, numCorr,
            position_d, num_points, scan, radius, maxPoints);
    CHECK_ERROR_MSG("Kernel Error");
}

void calculateAlignedPositions(float *result_pos_d, float *align_pos_d, float *position_d,
                               const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                               cublasHandle_t cnpHandle){
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
}

void calculatePointPairLoss(float *residual,
                            float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                            const PointPair point_pair, int num_resuduals,
                            const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                            const float weight, const bool isJacobianRequired){

    float *residual_d, *fa1Jacobian_d, *fa2Jacobian_d, *ftJacobian_d, *fuJacobian_d;

    // TODO: Do cuda malloc once ever (reuse) and memcopy of jacobians only when computing jacobians
    /*
    * Allocate residual to GPU
    */
    CUDA_ALLOC_AND_ZERO(&residual_d, static_cast<size_t >(num_resuduals));

    if (point_pair.point_count > 0) {
        calc_residual_point_pair(residual_d, point_pair, weight);
    }

    if (isJacobianRequired) {
        // Compute Jacobians for each parameter
        CUDA_ALLOC_AND_ZERO(&fa1Jacobian_d, static_cast<size_t >(num_resuduals*params.numa1));
        CUDA_ALLOC_AND_ZERO(&fa2Jacobian_d,  static_cast<size_t >(num_resuduals*params.numa2));
        CUDA_ALLOC_AND_ZERO(&ftJacobian_d, static_cast<size_t >(num_resuduals*params.numt));
        CUDA_ALLOC_AND_ZERO(&fuJacobian_d, static_cast<size_t >(num_resuduals*params.numu));

        if (point_pair.point_count > 0) {
            calc_derivatives_point_pair(ftJacobian_d, fuJacobian_d, fa1Jacobian_d, fa2Jacobian_d,
                                        params.fuParams_d, deformModel, point_pair, weight);
        }

        CUDA_CHECK(cudaMemcpy(fa1Jacobian, fa1Jacobian_d, num_resuduals*params.numa1 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(fa2Jacobian, fa2Jacobian_d, num_resuduals*params.numa2 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ftJacobian, ftJacobian_d, num_resuduals*params.numt * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(fuJacobian, fuJacobian_d, num_resuduals*params.numu * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(fa1Jacobian_d));
        CUDA_CHECK(cudaFree(fa2Jacobian_d));
        CUDA_CHECK(cudaFree(ftJacobian_d));
        CUDA_CHECK(cudaFree(fuJacobian_d));
    }

    /*
     * Copy computed residual and jacobian to Host
     */
    CUDA_CHECK(cudaMemcpy(residual, residual_d, num_resuduals*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(residual_d));
}

void calculateLandmarkLoss(float *residual, float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                           float *position_d, cublasHandle_t cnpHandle,
                           const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                           const float weight, const bool isJacobianRequired) {

    float *align_pos_d, *result_pos_d;
    float align_pos[deformModel.dim];


    // Allocate memory for Rigid aligned positions
    CUDA_CHECK(cudaMalloc((void **) &align_pos_d, deformModel.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &result_pos_d, deformModel.dim * sizeof(float)));
    // CuUDA Kernels run synchronously by default, to run asynchronously must explicitly specify streams

    // Calculate aligned positions
    calculateAlignedPositions(result_pos_d, align_pos_d, position_d, params, deformModel, scanPointCloud, cnpHandle);

    /*
     * Compute Point Pairs (Correspondances)
     */
    PointPair point_pair{
            .mesh_position_d=result_pos_d,
            .mesh_positoin_before_transform_d=align_pos_d,
            .ref_position_d=scanPointCloud.scanLandmark_d,
            .mesh_corr_inds_d=nullptr,
            .ref_corr_inds_d=nullptr,
            .point_count=scanPointCloud.numLmks
    };
    CUDA_MALLOC(&point_pair.mesh_corr_inds_d, static_cast<size_t>(scanPointCloud.numLmks));
    CUDA_MALLOC(&point_pair.ref_corr_inds_d, static_cast<size_t>(scanPointCloud.numLmks));

    calculateLandmarkIndices<<<1,scanPointCloud.numLmks>>>
                                 (point_pair.mesh_corr_inds_d, point_pair.ref_corr_inds_d, deformModel, scanPointCloud);

    // Calculate residual & jacobian for Landmarks
    calculatePointPairLoss(residual, fa1Jacobian, fa2Jacobian, ftJacobian, fuJacobian,
                           point_pair, point_pair.point_count*3,
                           params, deformModel, scanPointCloud,
                           weight, isJacobianRequired);

    CUDA_CHECK(cudaMemcpy(align_pos, align_pos_d, deformModel.dim * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(align_pos_d));
    CUDA_CHECK(cudaFree(result_pos_d));
    CUDA_FREE(point_pair.mesh_corr_inds_d);
    CUDA_FREE(point_pair.ref_corr_inds_d);
}

void calculateGeometricLoss(float *residual, float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                           float *position_d, cublasHandle_t cnpHandle,
                           const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                           const float searchRadius, const float weight, const int num_residuals, const bool isJacobianRequired) {
    float *align_pos_d, *result_pos_d;
    float align_pos[deformModel.dim];


    // Allocate memory for Rigid aligned positions
    CUDA_CHECK(cudaMalloc((void **) &align_pos_d, deformModel.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &result_pos_d, deformModel.dim * sizeof(float)));
    // CuUDA Kernels run synchronously by default, to run asynchronously must explicitly specify streams

    /*
     * Compute Loss
     */
    // Calculate aligned positions
    calculateAlignedPositions(result_pos_d, align_pos_d, position_d, params, deformModel, scanPointCloud, cnpHandle);

    /*
     * Compute Point Pairs (Correspondances)
     */
    PointPair point_pair{
            .mesh_position_d=result_pos_d,
            .mesh_positoin_before_transform_d=align_pos_d,
            .ref_position_d=scanPointCloud.scanPoints_d,
            .mesh_corr_inds_d=nullptr,
            .ref_corr_inds_d=nullptr,
            .point_count=0
    };

    float* distance_d;
    int* numCorr_d;
    float radius = searchRadius;

    CUDA_MALLOC(&point_pair.mesh_corr_inds_d, static_cast<size_t>(num_residuals));
    CUDA_MALLOC(&point_pair.ref_corr_inds_d, static_cast<size_t>(num_residuals));
    CUDA_MALLOC(&distance_d, static_cast<size_t>(num_residuals));
    CUDA_MALLOC(&numCorr_d, static_cast<size_t>(1));

    find_mesh_to_scan_corr(point_pair.mesh_corr_inds_d, point_pair.ref_corr_inds_d, distance_d, numCorr_d,
                           result_pos_d, deformModel.dim, scanPointCloud, radius, num_residuals);

    CUDA_CHECK(cudaMemcpy(&point_pair.point_count, numCorr_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (point_pair.point_count > num_residuals/3){
        point_pair.point_count = num_residuals/3;
    }

    /*******************
     * Calculate residual & jacobian for PointPairs
     *******************/
    calculatePointPairLoss(residual, fa1Jacobian, fa2Jacobian, ftJacobian, fuJacobian,
                           point_pair, num_residuals,
                           params, deformModel, scanPointCloud,
                           weight, isJacobianRequired);

    CUDA_CHECK(cudaMemcpy(align_pos, align_pos_d, deformModel.dim * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(align_pos_d));
    CUDA_CHECK(cudaFree(result_pos_d));
    CUDA_CHECK(cudaFree(distance_d));
    CUDA_CHECK(cudaFree(numCorr_d));
    CUDA_FREE(point_pair.mesh_corr_inds_d);
    CUDA_FREE(point_pair.ref_corr_inds_d);
}
