#include "face/cu_model_kernel.h"

#define BLOCKSIZE 128

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

void calculateLoss(float *residual, float *jacobian,
                   const float *position_d, const C_Params params,
                   const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud) {

}
