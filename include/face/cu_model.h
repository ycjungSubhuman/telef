#pragma once

#include <vector>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/** Each struct resides in host memory, while pointer members live in device memory */
typedef struct C_PcaDeformModel {
    float *deformBasis_d;
    float *ref_d;
    int *lmks_d;
    int lmkCount;
    int rank;
    int dim;
} C_PcaDeformModel;

typedef struct C_ScanPointCloud {
    float *scanPoints_d;
    int numPoints;
} C_ScanPointCloud;

typedef struct C_Params {
    float *params_d;
    int numParams;
} C_Params;

/** Loads MorphableFaceModel to GPU Device */
void loadModelToCUDADevice(C_PcaDeformModel *deformModel,
                           const Eigen::MatrixXf deformBasis, const Eigen::VectorXf ref,
                           const std::vector<int> lmkInds);

void freeModelCUDA(C_PcaDeformModel deformModel);

/** Loads PointCloud to GPU Device */
void loadScanToCUDADevice(C_ScanPointCloud *scanPointCloud,
                          std::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB>> scan);

void freeScanCUDA(C_ScanPointCloud scanPointCloud);

/** Loads PCA+Transform coefficients to GPU Device
 *
 * If update is true, doesn't malloc. uses previously allocated address
 */
void loadParamsToCUDADevice(C_Params *params, const float * const paramsIn, int numParams, bool update);

void freeParamsCUDA(C_Params *params);


/** Calculate vertex position given basis and coefficients */
void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);

/**
 * Calculate residual and jacobian of the loss function representing distance btw scan and model
 *
 * Loss = (L2 distance btw corresponding landmarks)
 *      + (L2 norm of parameters)
 */
void calculateLoss(float *residual, float *jacobian,
                   const float *position_d, const C_Params params,
                   const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud);
