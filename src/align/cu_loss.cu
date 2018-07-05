#include <device_launch_parameters.h>
#include <math.h>
#include "align/cu_loss.h"
#include "util/cu_quaternion.h"
#include "util/cu_reduce.h"
#include "util/cu_array.h"
#include "util/cudautil.h"

#define KERNEL_SIZE 512

/**
 * Calculate error^exponent of each element of each point, resulting in 3XN mse_cache_d
 *
 * error = (scan - position)
 */
__global__
void _calc_error_exp_cache_lmk(float *mse_cache_d, const float *position_d,
                               C_ScanPointCloud scan, float exponent) {

    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = scan.numLmks * 3;
    const int step = blockDim.x * gridDim.x;
    for(int i=start; i<size; i+=step) {
        const int lmk_idx = i/3;
        const int elem_idx = i%3;
        const int scan_point_idx = scan.scanLmks_d[lmk_idx];
        const int mesh_point_idx = scan.validModelLmks_d[lmk_idx];
        const float m_ik = scan.scanPoints_d[scan_point_idx*3 + elem_idx];
        const float x_m_ik = position_d[mesh_point_idx*3 + elem_idx];

        const float error_squared = powf(m_ik - x_m_ik, exponent);
        mse_cache_d[i] = error_squared;
    }
}

static void calc_error_exp_cache_lmk(float *mse_d, const float *position_d, C_ScanPointCloud scan, float exponent) {
    // This is used as cache before sum reduction
    float *error_exp_cache_d;
    CUDA_CHECK(cudaMalloc((void**)(&error_exp_cache_d), 3*scan.numLmks*sizeof(float)));
    _calc_error_exp_cache_lmk<<<1, scan.numLmks*3>>>(error_exp_cache_d, position_d, scan, exponent);

    // Reduce sum to get MSE
    const int numReductionThread = ((scan.numLmks*3 + 31) / 32) * 32;
    deviceReduceKernel<<<1, numReductionThread>>>(error_exp_cache_d, mse_d, scan.numLmks*3);
    CHECK_ERROR_MSG("Kernel Error");
    CUDA_CHECK(cudaFree(error_exp_cache_d));
}

static void calc_de_dt_lmk(float *de_dt_d, const float *error_cache_d, int num_point) {
    const int numReductionThread = ((num_point + 31) / 32) * 32;
    // error_cache_d is an array of x y z x y z x y z ...
    // We calculate sums for x components, y component, z component
    // and save it to de_dt_d[0], de_dt_d[1], de_dt_d[2] respectively
    deviceReduceKernelStrideScaled<<<1, numReductionThread>>>(error_cache_d, de_dt_d, num_point, 3, -2.0f);
    deviceReduceKernelStrideScaled<<<1, numReductionThread>>>(error_cache_d+1, de_dt_d+1, num_point, 3, -2.0f);
    deviceReduceKernelStrideScaled<<<1, numReductionThread>>>(error_cache_d+2, de_dt_d+2, num_point, 3, -2.0f);
    CHECK_ERROR_MSG("Kernel Error");
}

__global__
static void _calc_dx_m_du_lmk(float *dx_m_du, const float *u_d, const float *position_d, int num_points, C_ScanPointCloud scan) {
    float dr_du[27];
    calc_dr_du(dr_du, u_d);

    float *dx_m_duj[3] = {
            dx_m_du,
            dx_m_du + 3*num_points,
            dx_m_du + 2*3*num_points
    };

    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = num_points * 3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = 3;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j=y_start; j<y_size; j+=y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            float sum = 0.0f;

            sum += dr_du[9*j + 3*0 + i] * position_d[3*scan.validModelLmks_d[k] + 0];
            sum += dr_du[9*j + 3*1 + i] * position_d[3*scan.validModelLmks_d[k] + 1];
            sum += dr_du[9*j + 3*2 + i] * position_d[3*scan.validModelLmks_d[k] + 2];

            dx_m_duj[j][ind] = sum;
        }
    }
}

static void calc_de_du_lmk(float *de_du_d,
                           const float *error_cache_d, const float *u_d,
                           const float *position_d, C_ScanPointCloud scan) {
    float *dx_m_du; // concatenation of (dx_m_du1, dx_m_du2, dx_m_du3)
    CUDA_CHECK(cudaMalloc((void**)(&dx_m_du), 3*3*scan.numLmks*sizeof(float)));
    dim3 dimThread = dim3(static_cast<unsigned int>(3 * scan.numLmks), 3);
    _calc_dx_m_du_lmk<<<1, dimThread>>>(dx_m_du, u_d, position_d, scan.numLmks, scan);

    repeatedLinearSum(error_cache_d, dx_m_du, de_du_d, 3*scan.numLmks, 3);
    scale_array<<<1,3>>>(de_du_d, 3, -2.0f);
    CHECK_ERROR_MSG("Kernel Error");

    CUDA_CHECK(cudaFree(dx_m_du));
}

__global__
static void _calc_dx_da_lmk(float *dx_m_da, const float *u_d, int num_points, C_PcaDeformModel model, C_ScanPointCloud scan) {
    float r[9];
    calc_r_from_u(r, u_d);

    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = num_points * 3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = model.rank;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j=y_start; j<y_size; j+=y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            float sum = 0.0f;

            sum += r[0*3 + i] * model.deformBasis_d[3*scan.validModelLmks_d[k] + 0];
            sum += r[1*3 + i] * model.deformBasis_d[3*scan.validModelLmks_d[k] + 1];
            sum += r[2*3 + i] * model.deformBasis_d[3*scan.validModelLmks_d[k] + 2];

            dx_m_da[j*(num_points*3) + k*3 + i] = sum;
        }
    }
}

static void calc_de_da_lmk(float *de_da_d,
                           const float *error_cache_d, const float *u_d,
                           C_PcaDeformModel model, C_ScanPointCloud scan) {
    float *dx_m_da; // concatenation of (dx_m_da1, dx_m_da2, ..., dx_m_daN)
    CUDA_CHECK(cudaMalloc((void**)(&dx_m_da), model.rank*3*scan.numLmks*sizeof(float)));

    const int numThread = 32;
    const int xRequired = scan.numLmks * 3;
    const int yRequired = model.rank;
    dim3 dimBlock((xRequired + numThread - 1) / numThread, (yRequired + numThread - 1) / numThread);
    dim3 dimThread(numThread, numThread);
    _calc_dx_da_lmk<<<dimBlock, dimThread>>>(dx_m_da, u_d, scan.numLmks, model, scan);
    CHECK_ERROR_MSG("Kernel Error");

    repeatedLinearSum(error_cache_d, dx_m_da, de_da_d, 3*scan.numLmks, model.rank);
    scale_array<<<1, model.rank>>>(de_da_d, model.rank, -2.0f);
    CHECK_ERROR_MSG("Kernel Error");

    CUDA_CHECK(cudaFree(dx_m_da));
}

void calc_mse_lmk(float *mse_d, const float *position_d, C_ScanPointCloud scan) {
    calc_error_exp_cache_lmk(mse_d, position_d, scan, 2.0f);
    scale_array<<<1,1>>>(mse_d, 1, 1.0f/scan.numLmks);
    CHECK_ERROR_MSG("Kernel Error");
}


void calc_derivatives_lmk(float *de_dt_d, float *de_du_d, float *de_da_d,
                          const float *u_d, const float *position_before_tarnsform_d, const float *position_d,
                          C_PcaDeformModel model, C_ScanPointCloud scan) {
    float *error_cache_d;
    CUDA_CHECK(cudaMalloc((void**)(&error_cache_d), 3*scan.numLmks*sizeof(float)));
    _calc_error_exp_cache_lmk<<<1, 3*scan.numLmks>>>(error_cache_d, position_d, scan, 1.0f);
    CHECK_ERROR_MSG("Kernel Error");

    calc_de_dt_lmk(de_dt_d, error_cache_d, scan.numLmks);
    scale_array<<<1,3>>>(de_dt_d, 3, 1.0f/scan.numLmks);
    CHECK_ERROR_MSG("Kernel Error");
    calc_de_du_lmk(de_du_d, error_cache_d, u_d, position_before_tarnsform_d, scan);
    scale_array<<<1,3>>>(de_du_d, 3, 1.0f/scan.numLmks);
    CHECK_ERROR_MSG("Kernel Error");
    calc_de_da_lmk(de_da_d, error_cache_d, u_d, model, scan);
    scale_array<<<1,model.rank>>>(de_da_d, model.rank, 1.0f/scan.numLmks);
    CHECK_ERROR_MSG("Kernel Error");

    CUDA_CHECK(cudaFree(error_cache_d));
}
