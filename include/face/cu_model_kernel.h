#pragma once

#include <climits>
#include <float.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "align/cu_loss.h"
#include "face/raw_model.h"


const int NO_CORRESPONDENCE_I = INT_MAX;
const float INF_F = FLT_MAX;

__global__
void _calculateVertexPosition(float *position_d, const float *fa1Params_d, const float *fa2Params_d,
                              const C_PcaDeformModel deformModel);

void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);
void calculateVertexPosition(float *position_d, const float *fa1Params_d, const float *fa2Params_d, const C_PcaDeformModel deformModel);


__global__
void _homogeneousPositions(float *h_position_d, const float *position_d, int nPoints);

__global__
void _hnormalizedPositions(float *position_d, const float *h_position_d, int nPoints);

__global__
void _computeTransFromQ(float *trans, const float *u, const float *t);

//
//__device__
//void convertXyzToUv(float *uv, const float* xyz, float fx, float fy, float cx, float cy);

__global__
void _find_mesh_to_scan_corr(int *meshCorr_d, int *scanCorr_d, float *distance_d, int *numCorr,
                             const float *position_d, int num_points, C_ScanPointCloud scan, float radius=0, int maxPoints=0);

/**
 *
 * @param meshToScanCorr_d, Output Mesh To Scan Correspondance
 * @param distance_d, Output Corresponding Mesh to Scan Distance
 * @param position_d, Aligned Mesh
 * @param num_points
 * @param scan
 * @param radius, Tolerance or search window, if radius is 0, include all points
 */
void find_mesh_to_scan_corr(int *meshCorr_d, int *scanCorr_d, float *distance_d, int *numCorr,
                           const float *position_d, int num_points, C_ScanPointCloud scan, float radius=0, int maxPoints=0);
void reduce_closest_corr(int *meshCorr_d, int *scanCorr_d, float *distance_d, int *numCorr, int maxPoints);
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
 * Calculates Positions from deformModel and alignes them to the scan
 * @param result_pos_d
 * @param align_pos_d
 * @param position_d
 * @param params
 * @param deformModel
 * @param scanPointCloud
 * @param cnpHandle
 */
void calculateAlignedPositions(float *result_pos_d, float *align_pos_d, float *position_d,
                               const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                               cublasHandle_t cnpHandle);

void calculateAlignedPositionsCuda(float *result_pos_d, float *align_pos_d, float *position_d,
                                   const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                                   cublasHandle_t cnpHandle);

void calculatePointPairs(PointPair &point_pair, float *position_d, cublasHandle_t cnpHandle, C_Params params,
                         C_PcaDeformModel deformModel,
                         C_ScanPointCloud scanPointCloud);

/**
 * Calculate residual and jacobian of the loss function representing distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding PointPairs)
 *      + (L2 norm of parameters)
 */
void calculatePointPairLoss(float *residual, float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                            PointPair point_pair, C_Params params, C_PcaDeformModel deformModel,
                            C_Residuals c_residuals, C_Jacobians c_jacobians,
                            const float weight, const bool isJacobianRequired);

void calculatePointPairLossCuda(C_Residuals c_residuals, PointPair point_pair, const float weight);
void calculatePointPairDerivatives(C_Jacobians c_jacobians, PointPair point_pair, C_Params params, C_PcaDeformModel deformModel,
                                   const float weight);

/**
 * Calculate residual and jacobian of the loss function representing distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding Landmark)
 *      + (L2 norm of parameters)
 */
void calculateLandmarkLossCuda(float *position_d, cublasHandle_t cnpHandle, C_Params params, C_PcaDeformModel deformModel,
                               C_ScanPointCloud scanPointCloud, C_Residuals c_residuals, C_Jacobians c_jacobians,
                               const float weight, const bool isJacobianRequired);

/**
 * Calculate residual and jacobian of the loss function representing distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding Landmark)
 *      + (L2 norm of parameters)
 */
void calculateLandmarkLoss(float *residual, float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                           float *position_d, cublasHandle_t cnpHandle, C_Params params, C_PcaDeformModel deformModel,
                           C_ScanPointCloud scanPointCloud, C_Residuals c_residuals, C_Jacobians c_jacobians,
                           const float weight, const bool isJacobianRequired);

/**
 * Calculate residual and jacobian of the loss function representing distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding Mesh and Scan points)
 *      + (L2 norm of parameters)
 */
void calculateGeometricLoss(float *residual, float *fa1Jacobian, float *fa2Jacobian, float *ftJacobian, float *fuJacobian,
                            float *position_d, cublasHandle_t cnpHandle, const C_Params params,
                            const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                            C_Residuals c_residuals, C_Jacobians c_jacobians, const float searchRadius, const float weight,
                            const bool isJacobianRequired);