#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "face/raw_model.h"

__global__
void _calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);

void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);

__global__
void _homogeneousPositions(float *h_position_d, const float *position_d, int nPoints);

__global__
void _hnormalizedPositions(float *position_d, const float *h_position_d, int nPoints);

/**
 * Applies Transformation matrix on CUDA device model
 * @param align_pos_d
 * @param position_d
 * @param deformModel
 * @param scanPointCloud
 */
void applyRigidAlignment(float *align_pos_d, cublasHandle_t cnpHandle,
                         const float *position_d, const float *transMat, int N);

/**
 * GPU MatrixMultiply using Cublas
 * @param matC
 * @param matA_host
 * @param aCols
 * @param aRows
 * @param matB
 * @param bCols
 * @param bRows
 */
void cudaMatMul(float *matC, cublasHandle_t cnpHandle,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols);

/**
 * Calculate residual and jacobian of the loss function representing distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding landmarks)
 *      + (L2 norm of parameters)
 */
void calculateLoss(float *residual, float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                   float *position_d,
                   cublasHandle_t cnpHandle,
                   const C_Params params,
                   const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                   const bool isJacobianRequired);