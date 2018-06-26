#pragma once

#include <vector>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/* Includes, cuda */
#include <cuda_runtime.h>
//#include <cublas_v2.h>


#include "face/raw_model.h"


/** Loads MorphableFaceModel to GPU Device */
void loadModelToCUDADevice(C_PcaDeformModel *deformModel,
                           const Eigen::MatrixXf deformBasis, const Eigen::VectorXf ref,
                           const std::vector<int> lmkInds);
void freeModelCUDA(C_PcaDeformModel deformModel);

/** Loads PointCloud to GPU Device */
void loadScanToCUDADevice(C_ScanPointCloud *scanPointCloud,
                          boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA>> scan,
                          const std::vector<int> scanLmkIdx,
                          const std::vector<int> validLmks,
                          const Eigen::MatrixXf rigidTransform);

void freeScanCUDA(C_ScanPointCloud scanPointCloud);

/**
 * Loads PCA+Transform coefficients to GPU Device
 *
 * If update is true, doesn't malloc. uses previously allocated address
 */
void allocParamsToCUDADevice(C_Params *params, int numParams);
void updateParamsInCUDADevice(const C_Params params, const float * const paramsIn, int numParams);
void freeParamsCUDA(C_Params params);

void allocPositionCUDA(float **position_d, int dim);
void freePositionCUDA(float *position_d);

/** Calculate vertex position given basis and coefficients */
void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);

/**
 * Calculate residual and jacobian of the loss function representing Landmark distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding landmarks)
 *      + (L2 norm of parameters)
 */
void calculateLandmarkLoss(float *residual, float *jacobian,
                           const float *position_d, const C_Params params,
                           const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud);

/**
 * Applies Transformation matrix on CUDA device model
 * @param align_pos_d
 * @param position_d
 * @param deformModel
 * @param scanPointCloud
 */
void applyRigidAlignment(float *align_pos_d, const float *position_d,
                         const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud);

/**
 * Calculate residual and jacobian of the loss function representing distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding landmarks)
 *      + (L2 norm of parameters)
 */
void calculateLoss(float *residual, float *jacobian,
                   const float *position_d, const C_Params params,
                   const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                   const bool isJacobianRequired);
