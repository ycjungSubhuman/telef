#include <cuda_runtime_api.h>

#include "util/cudautil.h"
#include "face/model_cudahelper.h"


void loadModelToCUDADevice(C_PcaDeformModel *deformModel,
                           const Eigen::MatrixXf shapeDeformBasis, const Eigen::MatrixXf expressionDeformBasis,
                           const Eigen::VectorXf ref,
                           const Eigen::VectorXf meanShapeDeformation,
                           const Eigen::VectorXf meanExpressionDeformation,
                           const std::vector<int> lmkInds) {
    //std::cout << "shapeDeformBasis.size() <<: " << shapeDeformBasis.size() << std::endl;
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->shapeDeformBasis_d), shapeDeformBasis.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->expressionDeformBasis_d), expressionDeformBasis.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->ref_d), ref.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->meanShapeDeformation_d),
                          meanShapeDeformation.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->meanExpressionDeformation_d),
                          meanExpressionDeformation.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->lmks_d), lmkInds.size()*sizeof(int)));

    CUDA_CHECK(cudaMemcpy((void*)deformModel->shapeDeformBasis_d,
               shapeDeformBasis.data(), shapeDeformBasis.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->expressionDeformBasis_d,
                          expressionDeformBasis.data(), expressionDeformBasis.size()*sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->ref_d,
               ref.data(), ref.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->meanShapeDeformation_d,
               meanShapeDeformation.data(), meanShapeDeformation.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->meanExpressionDeformation_d,
                          meanExpressionDeformation.data(), meanExpressionDeformation.size()*sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->lmks_d,
               lmkInds.data(), lmkInds.size()*sizeof(int), cudaMemcpyHostToDevice));

    deformModel->shapeRank = (int)shapeDeformBasis.cols();
    deformModel->expressionRank = (int)expressionDeformBasis.cols();
    deformModel->dim = (int)ref.size();
    deformModel->lmkCount = (int)lmkInds.size();

    assert(shapeDeformBasis.rows() == ref.size());
}

void freeModelCUDA(C_PcaDeformModel deformModel) {
    CUDA_CHECK(cudaFree(deformModel.shapeDeformBasis_d));
    CUDA_CHECK(cudaFree(deformModel.expressionDeformBasis_d));
    CUDA_CHECK(cudaFree(deformModel.meanExpressionDeformation_d));
    CUDA_CHECK(cudaFree(deformModel.meanShapeDeformation_d));
    CUDA_CHECK(cudaFree(deformModel.ref_d));
    CUDA_CHECK(cudaFree(deformModel.lmks_d));
}

void loadScanToCUDADevice(C_ScanPointCloud *scanPointCloud,
                          boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA>> scan,
                          float fx, float fy,
                          const std::vector<int> scanLmkIdx,
                          const std::vector<int> validLmks,
                          const Eigen::MatrixXf rigidTransform) {

    CUDA_CHECK(cudaMalloc((void**)(&scanPointCloud->scanPoints_d), scan->points.size()*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&scanPointCloud->validModelLmks_d), validLmks.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)(&scanPointCloud->scanLmks_d), scanLmkIdx.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)(&scanPointCloud->rigidTransform_d), rigidTransform.size()*sizeof(float)));

    float *scanPoints = new float[scan->points.size()*3];
    for (int i=0; i<scan->points.size(); i++) {
        scanPoints[3*i] = scan->points[i].x;
        scanPoints[3*i+1] = scan->points[i].y;
        scanPoints[3*i+2] = scan->points[i].z;
    }

    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->scanPoints_d,
               scanPoints, scan->points.size()*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->validModelLmks_d,
               validLmks.data(), validLmks.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->scanLmks_d,
               scanLmkIdx.data(), scanLmkIdx.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->rigidTransform_d,
               rigidTransform.data(), rigidTransform.size()*sizeof(float), cudaMemcpyHostToDevice));

    scanPointCloud->width = scan->width;
    scanPointCloud->height = scan->height;
    scanPointCloud->fx = fx;
    scanPointCloud->fy = fy;
    scanPointCloud->cx = (static_cast<float>(scan->width) - 1.f) / 2.f;
    scanPointCloud->cy = (static_cast<float>(scan->height) - 1.f) / 2.f;

    scanPointCloud->numPoints = static_cast<int>(scan->points.size());
    scanPointCloud->transformCols = (int)rigidTransform.cols();
    scanPointCloud->transformRows = (int)rigidTransform.rows();
    scanPointCloud->numLmks = static_cast<int>(scanLmkIdx.size());

    assert(scanLmkIdx.size() == validLmks.size());
}

void freeScanCUDA(C_ScanPointCloud scanPointCloud) {
    cudaFree(scanPointCloud.scanPoints_d);
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
    CUDA_CHECK(cudaFree(params.fa1Params_d));
    CUDA_CHECK(cudaFree(params.fa2Params_d));
    CUDA_CHECK(cudaFree(params.ftParams_d));
    CUDA_CHECK(cudaFree(params.fuParams_d));

    delete[] params.ftParams_h;
    delete[] params.fuParams_h;
}

void allocPositionCUDA(float **position_d, int dim) {
    CUDA_CHECK(cudaMalloc((void**)(position_d), dim*sizeof(float)));
}

void freePositionCUDA(float *position_d) {
    CUDA_CHECK(cudaFree(position_d));
}

