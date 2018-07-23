#include <cuda_runtime_api.h>

#include "util/cudautil.h"
#include "face/model_cudahelper.h"
#include "type.h"

namespace {
    using namespace telef::types;

    void copyPointCloudPosition(float *position, CloudConstPtrT cloud) {
        for (int i=0; i<cloud->points.size(); i++) {
            position[3*i] = cloud->points[i].x;
            position[3*i+1] = cloud->points[i].y;
            position[3*i+2] = cloud->points[i].z;
        }
    }
}


void loadModelToCUDADevice(C_PcaDeformModel *deformModel,
                           const Eigen::MatrixXf shapeDeformBasis, const Eigen::MatrixXf expressionDeformBasis,
                           const Eigen::VectorXf ref,
                           const Eigen::VectorXf shapeDeformationCenter,
                           const Eigen::VectorXf expressionDeformationCenter,
                           const std::vector<int> lmkInds) {
    CUDA_ALLOC_AND_COPY(&deformModel->shapeDeformBasis_d, shapeDeformBasis.data(),
                        static_cast<size_t>(shapeDeformBasis.size()));
    CUDA_ALLOC_AND_COPY(&deformModel->expressionDeformBasis_d, expressionDeformBasis.data(),
                        static_cast<size_t>(expressionDeformBasis.size()));
    CUDA_ALLOC_AND_COPY(&deformModel->ref_d, ref.data(),
                        static_cast<size_t>(ref.size()));
    CUDA_ALLOC_AND_COPY(&deformModel->meanShapeDeformation_d, shapeDeformationCenter.data(),
                        static_cast<size_t>(shapeDeformationCenter.size()));
    CUDA_ALLOC_AND_COPY(&deformModel->meanExpressionDeformation_d, expressionDeformationCenter.data(),
                        static_cast<size_t>(expressionDeformationCenter.size()));
    CUDA_ALLOC_AND_COPY(&deformModel->lmks_d, lmkInds.data(), lmkInds.size());

    deformModel->shapeRank = (int)shapeDeformBasis.cols();
    deformModel->expressionRank = (int)expressionDeformBasis.cols();
    deformModel->dim = (int)ref.size();
    deformModel->lmkCount = (int)lmkInds.size();

    assert(shapeDeformBasis.rows() == ref.size());
}

void freeModelCUDA(C_PcaDeformModel deformModel) {
    CUDA_FREE(deformModel.shapeDeformBasis_d);
    CUDA_FREE(deformModel.expressionDeformBasis_d);
    CUDA_FREE(deformModel.meanExpressionDeformation_d);
    CUDA_FREE(deformModel.meanShapeDeformation_d);
    CUDA_FREE(deformModel.ref_d);
    CUDA_FREE(deformModel.lmks_d);
}

void loadScanToCUDADevice(C_ScanPointCloud *scanPointCloud,
                          boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA>> scan,
                          float fx, float fy,
                          std::vector<int> modelLandmarkSelection,
                          Eigen::Matrix4f rigidTransform, CloudConstPtrT landmark3d) {
    float *scanPoints = new float[scan->points.size()*3];
    float *scanLandmarks = new float[landmark3d->points.size()*3];
    copyPointCloudPosition(scanPoints, scan);
    copyPointCloudPosition(scanLandmarks, landmark3d);

    CUDA_ALLOC_AND_COPY(&scanPointCloud->scanPoints_d, scanPoints, scan->points.size()*3);
    CUDA_ALLOC_AND_COPY(&scanPointCloud->rigidTransform_d, rigidTransform.data(),
                        static_cast<size_t>(rigidTransform.size()));
    CUDA_ALLOC_AND_COPY(&scanPointCloud->scanLandmark_d, scanLandmarks, landmark3d->points.size()*3);
    CUDA_ALLOC_AND_COPY(&scanPointCloud->modelLandmarkSelection_d, modelLandmarkSelection.data(),
                        modelLandmarkSelection.size());

    scanPointCloud->width = scan->width;
    scanPointCloud->height = scan->height;
    scanPointCloud->fx = fx;
    scanPointCloud->fy = fy;
    scanPointCloud->cx = (static_cast<float>(scan->width) - 1.f) / 2.f;
    scanPointCloud->cy = (static_cast<float>(scan->height) - 1.f) / 2.f;

    scanPointCloud->numPoints = static_cast<int>(scan->points.size());
    scanPointCloud->numLmks = static_cast<int>(modelLandmarkSelection.size());

    delete[] scanPoints;
    delete[] scanLandmarks;
}

void freeScanCUDA(C_ScanPointCloud scanPointCloud) {
    CUDA_FREE(scanPointCloud.scanPoints_d);
    CUDA_FREE(scanPointCloud.rigidTransform_d);
    CUDA_FREE(scanPointCloud.scanLandmark_d);
    CUDA_FREE(scanPointCloud.modelLandmarkSelection_d);
}

void allocParamsToCUDADevice(C_Params *params, int numa1, int numa2, int numt, int numu) {
    CUDA_CHECK(cudaMalloc((void **)(&params->fa1Params_d), numa1*sizeof(float)));
    float *zeroA1 = new float[numa1]{0,};
    params->numa1 = numa1;

    CUDA_CHECK(cudaMalloc((void **)(&params->fa2Params_d), numa2*sizeof(float)));
    float *zeroA2 = new float[numa2]{0,};
    params->numa2 = numa2;

    CUDA_CHECK(cudaMalloc((void **)(&params->ftParams_d), numt*sizeof(float)));
    params->ftParams_h = new float[numt]{0,};
    params->numt = numt;

    CUDA_CHECK(cudaMalloc((void **)(&params->fuParams_d), numu*sizeof(float)));
    params->fuParams_h = new float[numu]{0,};
    params->numu = numu;

    updateParams(*params, zeroA1, numa1, zeroA2, numa2, params->ftParams_h, numt, params->fuParams_h, numu);
    delete[] zeroA1;
    delete[] zeroA2;
}

void updateParams(const C_Params params,
                  const float *const a1In, int numa1,
                  const float *const a2In, int numa2,
                  const float *const tIn, int numt,
                  const float *const uIn, int numu) {
    CUDA_CHECK(cudaMemcpy((void*)params.fa1Params_d, a1In, numa1*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)params.fa2Params_d, a2In, numa2*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)params.ftParams_d, tIn, numt*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)params.fuParams_d, uIn, numu*sizeof(float), cudaMemcpyHostToDevice));

    memcpy((void*)params.ftParams_h, tIn, numt*sizeof(float));
    memcpy((void*)params.fuParams_h, uIn, numu*sizeof(float));
}

void freeParamsCUDA(C_Params params) {
    CUDA_FREE(params.fa1Params_d);
    CUDA_FREE(params.fa2Params_d);
    CUDA_FREE(params.ftParams_d);
    CUDA_FREE(params.fuParams_d);

    delete[] params.ftParams_h;
    delete[] params.fuParams_h;
}

void allocPositionCUDA(float **position_d, int dim) {
    CUDA_CHECK(cudaMalloc((void**)(position_d), dim*sizeof(float)));
}

void freePositionCUDA(float *position_d) {
    CUDA_FREE(position_d);
}

