#include <cuda_runtime_api.h>

#include "util/cudautil.h"
#include "face/model_cudahelper.h"


void loadModelToCUDADevice(C_PcaDeformModel *deformModel,
                           const Eigen::MatrixXf deformBasis, const Eigen::VectorXf ref,
                           const Eigen::VectorXf meanDeformation,
                           const std::vector<int> lmkInds) {
    //std::cout << "deformBasis.size() <<: " << deformBasis.size() << std::endl;
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->deformBasis_d), deformBasis.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->ref_d), ref.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->mean_d), meanDeformation.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->lmks_d), lmkInds.size()*sizeof(int)));

    CUDA_CHECK(cudaMemcpy((void*)deformModel->deformBasis_d,
               deformBasis.data(), deformBasis.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->ref_d,
               ref.data(), ref.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->mean_d,
               meanDeformation.data(), meanDeformation.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->lmks_d,
               lmkInds.data(), lmkInds.size()*sizeof(int), cudaMemcpyHostToDevice));

    deformModel->rank = (int)deformBasis.cols();
    deformModel->dim = (int)deformBasis.rows();
    deformModel->lmkCount = (int)lmkInds.size();

    assert(deformBasis.rows() == ref.size());
}

void freeModelCUDA(C_PcaDeformModel deformModel) {
    cudaFree(deformModel.deformBasis_d);
    cudaFree(deformModel.ref_d);
    cudaFree(deformModel.lmks_d);
    cudaFree(deformModel.mean_d);
}

void loadScanToCUDADevice(C_ScanPointCloud *scanPointCloud,
                          boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA>> scan,
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

    scanPointCloud->numPoints = static_cast<int>(scan->points.size());
    scanPointCloud->transformCols = (int)rigidTransform.cols();
    scanPointCloud->transformRows = (int)rigidTransform.rows();
    scanPointCloud->numLmks = static_cast<int>(scanLmkIdx.size());

    assert(scanLmkIdx.size() == validLmks.size());
}

void freeScanCUDA(C_ScanPointCloud scanPointCloud) {
    cudaFree(scanPointCloud.scanPoints_d);
}

void allocParamsToCUDADevice(C_Params *params, int numa, int numt, int numu) {
    CUDA_CHECK(cudaMalloc((void **)(&params->faParams_d), numa*sizeof(float)));
    float *zeroA = new float[numa]{0,};
    params->numa = numa;

    CUDA_CHECK(cudaMalloc((void **)(&params->ftParams_d), numt*sizeof(float)));
    params->ftParams_h = new float[numt]{0,};
    params->numt = numt;

    CUDA_CHECK(cudaMalloc((void **)(&params->fuParams_d), numu*sizeof(float)));
    params->fuParams_h = new float[numu]{0,};
    params->numu = numu;

    updateParams(*params, zeroA, numa, params->ftParams_h, numt, params->fuParams_h, numu);
    delete[] zeroA;
}

void updateParams(const C_Params params,
                  const float *const aIn, int numa,
                  const float *const tIn, int numt,
                  const float *const uIn, int numu) {
    CUDA_CHECK(cudaMemcpy((void*)params.faParams_d, aIn, numa*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)params.ftParams_d, tIn, numt*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)params.fuParams_d, uIn, numu*sizeof(float), cudaMemcpyHostToDevice));

    memcpy((void*)params.ftParams_h, tIn, numt*sizeof(float));
    memcpy((void*)params.fuParams_h, uIn, numu*sizeof(float));
}

void freeParamsCUDA(C_Params params) {
    CUDA_CHECK(cudaFree(params.faParams_d));
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

