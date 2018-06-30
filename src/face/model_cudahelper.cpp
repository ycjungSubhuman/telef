#include <cuda_runtime_api.h>

#include "util/cudautil.h"
#include "face/model_cudahelper.h"


void loadModelToCUDADevice(C_PcaDeformModel *deformModel,
                           const Eigen::MatrixXf deformBasis, const Eigen::VectorXf ref,
                           const std::vector<int> lmkInds) {
    //std::cout << "deformBasis.size() <<: " << deformBasis.size() << std::endl;
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->deformBasis_d), deformBasis.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->ref_d), ref.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&deformModel->lmks_d), lmkInds.size()*sizeof(int)));

    CUDA_CHECK(cudaMemcpy((void*)deformModel->deformBasis_d,
               deformBasis.data(), deformBasis.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)deformModel->ref_d,
               ref.data(), ref.size()*sizeof(float), cudaMemcpyHostToDevice));
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
    for (int i=0; i<scan->points.size(); i+=3) {
        scanPoints[i] = scan->points[i].x;
        scanPoints[i+1] = scan->points[i].y;
        scanPoints[i+2] = scan->points[i].z;
    }

    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->scanPoints_d,
               scanPoints, scan->points.size()*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->validModelLmks_d,
               validLmks.data(), validLmks.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->scanLmks_d,
               scanLmkIdx.data(), scanLmkIdx.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)scanPointCloud->rigidTransform_d,
               rigidTransform.data(), rigidTransform.size()*sizeof(float), cudaMemcpyHostToDevice));

    scanPointCloud->numPoints = scan->points.size();
    scanPointCloud->transformCols = (int)rigidTransform.cols();
    scanPointCloud->transformRows = (int)rigidTransform.rows();
    scanPointCloud->numLmks = scanLmkIdx.size();

    assert(scanLmkIdx.size() == validLmks.size());
}

void freeScanCUDA(C_ScanPointCloud scanPointCloud) {
    cudaFree(scanPointCloud.scanPoints_d);
}

void allocParamsToCUDADevice(C_Params *params, int numParams) {
    CUDA_CHECK(cudaMalloc((void **)(&params->params_d), numParams*sizeof(float)));
    float *zero = new float[numParams]{0,};
    params->numParams = numParams;

    updateParamsInCUDADevice(*params, zero, numParams);
    delete[] zero;
}

void updateParamsInCUDADevice(const C_Params params, const float * const paramsIn, int numParams) {
    CUDA_CHECK(cudaMemcpy((void*)params.params_d, paramsIn, numParams*sizeof(float), cudaMemcpyHostToDevice));
}

void freeParamsCUDA(C_Params params) {
    cudaFree(params.params_d);
}

void allocPositionCUDA(float **position_d, int dim) {
    CUDA_CHECK(cudaMalloc((void**)(position_d), dim*sizeof(float)));
}

void freePositionCUDA(float *position_d) {
    cudaFree(position_d);
}

