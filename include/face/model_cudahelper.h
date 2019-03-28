#pragma once

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

#include "face/raw_model.h"
#include "type.h"
#include <cuda_runtime.h>

namespace {
using namespace telef::types;
}

/** Loads MorphableFaceModel to GPU Device */
void loadModelToCUDADevice(
    C_PcaDeformModel *deformModel,
    const Eigen::MatrixXf shapeDeformBasis,
    const Eigen::MatrixXf expressionDeformBasis,
    const Eigen::VectorXf ref,
    const Eigen::VectorXf shapeDeformationCenter,
    const Eigen::VectorXf expressionDeformationCenter,
    const std::vector<int> lmkInds);
void freeModelCUDA(C_PcaDeformModel deformModel);

/** Loads PointCloud to GPU Device */
void loadScanToCUDADevice(
    C_ScanPointCloud *scanPointCloud,
    boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA>> scan,
    float fx,
    float fy,
    std::vector<int> modelLandmarkSelection,
    Eigen::Matrix4f rigidTransform,
    CloudConstPtrT landmark3d);

void freeScanCUDA(C_ScanPointCloud scanPointCloud);

/**
 * Loads PCA+Transform coefficients to GPU Device
 *
 * If update is true, doesn't malloc. uses previously allocated address
 */
void allocParamsToCUDADevice(
    C_Params *params, int numa1, int numa2, int numt, int numu);
void updateParams(
    const C_Params params,
    const float *const a1In,
    int numa1,
    const float *const a2In,
    int numa2,
    const float *const tIn,
    int numt,
    const float *const uIn,
    int numu);
void freeParamsCUDA(C_Params params);

void allocResidualsToCUDADevice(C_Residuals *residuals, int num_residuals);
void zeroResidualsCUDA(const C_Residuals residuals);
void freeResidualsCUDA(const C_Residuals residuals);

void allocJacobiansToCUDADevice(
    C_Jacobians *jacobians,
    int num_residuals,
    int numa1,
    int numa2,
    int numt,
    int numu);
void zeroJacobiansCUDA(const C_Jacobians jacobians);
void freeJacobiansCUDA(const C_Jacobians jacobians);

void allocPositionCUDA(float **position_d, int dim);
void freePositionCUDA(float *position_d);
