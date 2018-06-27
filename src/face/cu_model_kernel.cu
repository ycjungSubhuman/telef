#include "face/cu_model_kernel.h"

#include <iostream>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCKSIZE 128

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {

    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__
void _calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int colDim = deformModel.dim;

    position_d[i] = 0;
    for (int j=0; j<deformModel.rank; j++) {
        position_d[i] += params.params_d[j] * deformModel.deformBasis_d[i + colDim * j];
    }
}

void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel) {
    int idim = deformModel.dim;
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((idim + BLOCKSIZE-1)/BLOCKSIZE);
    _calculateVertexPosition<<<dimGrid, dimBlock>>>(position_d, params, deformModel);
}

__global__
void _calculateLandmarkLoss(float *residual_d, float *jacobian_d,
                   const float *position_d, const C_Params params,
                   const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                   const bool isJacobianRequired) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    //TODO: Add param for weights
    float landmarkCoeff = 1.0;
    const int colDim = deformModel.dim;

    // This Treads Placeholders for output
    float res = 0.0;

    for (int idx = index; idx < scanPointCloud.numLmks; idx += blockDim.x * gridDim.x) {
        int posIdx = deformModel.lmks_d[idx];
        int scanIdx = scanPointCloud.scanLmks_d[idx];


        float ptSubt[3] = { position_d[3 * posIdx] - scanPointCloud.scanPoints_d[3 * scanIdx],           // x
                            position_d[3 * posIdx + 1] - scanPointCloud.scanPoints_d[3 * scanIdx + 1],   // y
                            position_d[3 * posIdx + 2] - scanPointCloud.scanPoints_d[3 * scanIdx + 2] }; // z

        float squaredNorm = 0.0;

        for (int i = 0; i < 3; i++) {
            squaredNorm += ptSubt[i] * ptSubt[i];
        }

        res += landmarkCoeff * squaredNorm;

        if(isJacobianRequired) {
            for (int j=0; j<deformModel.rank; j++) {
                float basis[3] = { deformModel.deformBasis_d[3*index + colDim * j],       // x @ col j
                                   deformModel.deformBasis_d[3*index + 1 + colDim * j ],  // y @ col j
                                   deformModel.deformBasis_d[3*index + 2 + colDim * j] }; // z @ col j

                // Element wise multiplication and sum
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += ptSubt[k] * basis[k];
                }

                float jacobi = -2 * landmarkCoeff * sum;

                // Reduce Jacobians across across block
                jacobi = blockReduceSum(jacobi);

                // Add partial sum into atomic output, only do it once per block
                if (threadIdx.x == 0) {
                    atomicAdd(&jacobian_d[j], jacobi);
                }
            }
        }
    }

    // Reduce Residuals across block
    res = blockReduceSum(res);

    // Add partial sum into atomic output, only do it once per block
    if (threadIdx.x == 0) {
        atomicAdd(residual_d, res);
    }
}

void calculateLandmarkLoss(float *residual_d, float *jacobian_d,
                            const float *position_d, const C_Params params,
                            const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                            const bool isJacobianRequired) {
    int idim = scanPointCloud.numLmks;
    dim3 lmkThrds(BLOCKSIZE);
    dim3 lmkBlocks((idim/BLOCKSIZE) + 1);

    _calculateLandmarkLoss<<<lmkBlocks, lmkThrds>>>(residual_d, jacobian_d,
            position_d, params, deformModel, scanPointCloud, isJacobianRequired);
}

__global__
void _applyRigidAlignment(cublasStatus_t *d_status, float *align_pos_d, const float *position_d,
                          const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    cublasHandle_t cnpHandle;
    cublasStatus_t status = cublasCreate(&cnpHandle);

    // Don't know what this is (scalar?) but examples use this
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        *d_status = status;
        return;
    }

    // homogeneous coordinates (x,y,z,1);
    float pos[4] = {position_d[3 * index], position_d[3 * index + 1], position_d[3 * index + 2], 1};

    // homogeneous result
    float h_aligned[4];

    /* Perform operation using cublas, inputs/outputs are col-major.
     * vector and array were originally Eigen which defaults to Col-major
     * m is rows for A and C
     * n is cols for B and C
     * k is cols for A and rows for B*/
    status =
            cublasSgemm(cnpHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        4, 1, 4, //(m,n,k)
                        alpha,
                        scanPointCloud.rigidTransform_d, 4, //(4x4) or (mxk)
                        pos, 1, //(4x1) or (kxn)
                        beta,
                        h_aligned, 4); //(4x1) or (mxk)

    cublasDestroy(cnpHandle);

    // hnormalized point (x,y,z)
    memcpy(align_pos_d, h_aligned, 3*sizeof(int));

    *d_status = status;
}

void applyRigidAlignment(float *align_pos_d, const float *position_d,
                         const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud) {
    int idim = deformModel.dim/3;
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((idim + BLOCKSIZE-1)/BLOCKSIZE);
    cublasStatus_t *d_status;
    cublasStatus_t status;

    cudaMalloc((void **) &d_status, sizeof(cublasStatus_t));

    _applyRigidAlignment<<<dimGrid, dimBlock>>>(d_status, align_pos_d, position_d,
            deformModel, scanPointCloud);

    cudaMemcpy(&status, d_status, sizeof(cublasStatus_t), cudaMemcpyDeviceToHost);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "!!! CUBLAS Device API call failed with code %d\n" << std::endl;
    }

    cudaFree(d_status);


}

void calculateLoss(float *residual, float *jacobian, float *position_d,
                   const C_Params params, const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud,
                   const bool isJacobianRequired) {

    std::cout << "calculateLoss" << std::endl;
    float *residual_d, *jacobian_d;
    float *align_pos_d;

    /*
     * Allocate and Copy residual amd jacobian to GPU
     */
    // Currently, We are using only 1 residual to contain the loss
    cudaMalloc(&residual_d, sizeof(float));

    // Compute Jacobians for each parameter
    cudaMalloc((void**)&jacobian_d, params.numParams*sizeof(float));

    // Allocate memory for Rigid aligned positions
    cudaMalloc((void**)&align_pos_d, deformModel.dim*sizeof(float));

    cudaMemcpy(residual_d,  residual, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(jacobian_d,  jacobian, params.numParams*sizeof(float), cudaMemcpyHostToDevice);

    // CuUDA Kernels run synchronously by default, to run asynchronously must explicitly specify streams

    /*
     * Compute Loss
     */
    // Calculate position_d

    std::cout << "calculateLoss: calculateVertexPosition" << std::endl;
    calculateVertexPosition(position_d, params, deformModel);
    //cudaDeviceSynchronize();

    // Rigid alignment
    std::cout << "calculateLoss: applyRigidAlignment" << std::endl;
    applyRigidAlignment(align_pos_d, position_d, deformModel, scanPointCloud);
    //cudaDeviceSynchronize();

    // Calculate residual_d, jacobian_d for Landmarks
    std::cout << "calculateLoss: calculateLandmarkLoss" << std::endl;
    calculateLandmarkLoss(residual_d, jacobian_d,
                          align_pos_d, params, deformModel, scanPointCloud, isJacobianRequired);
    //cudaDeviceSynchronize();

    /*
     * Copy computed residual and jacobian to Host
     */
    std::cout << "Copy to host" << std::endl;
    cudaMemcpy(residual, residual_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(jacobian, jacobian_d, params.numParams*sizeof(float), cudaMemcpyDeviceToHost);

    //TODO: return value to see rigid aligned mesh?

    std::cout << "Free cuda" << std::endl;
    cudaFree(align_pos_d);
}
