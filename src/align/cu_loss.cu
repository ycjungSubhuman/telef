#include <device_launch_parameters.h>
#include <math.h>
#include "align/cu_loss.h"
#include "util/cu_quaternion.h"
#include "util/cu_array.h"
#include "util/cudautil.h"

/**
 * Calculate error of each element of each point, resulting in 3xN array
 *
 * error = (scan - position)
 */
__global__
static void _calc_error_lmk(float *error_d, const float *position_d,
                     C_ScanPointCloud scan) {

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

        const float error = m_ik - x_m_ik;
        error_d[i] = error;
    }
}

static void calc_error_lmk(float *error_d, const float *position_d, C_ScanPointCloud scan) {
    // This is used as cache before sum reduction
    _calc_error_lmk<<<1, scan.numLmks*3>>>(error_d, position_d, scan);
    CHECK_ERROR_MSG("Kernel Error");
}

__global__
static void _calc_dx_m_dt_lmk(float *dx_m_dt, int num_points) {
    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = num_points * 3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = 3;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j = y_start; j < y_size; j += y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            if(i==j) {
                dx_m_dt[3*3*k + 3*i + j] = -1.0f;
            }
            else {
                dx_m_dt[3*3*k + 3*i + j] = 0.0;
            }
        }
    }
}

static void calc_de_dt_lmk(float *de_dt_d, int num_points) {
    dim3 dimThread = dim3(static_cast<unsigned int>(3 * num_points), 3);
    _calc_dx_m_dt_lmk<<<1, dimThread>>>(de_dt_d, num_points);
    CHECK_ERROR_MSG("Kernel Error");
    scale_array<<<1,num_points*3*3>>>(de_dt_d, num_points*3*3, 1.0f/sqrtf(num_points));
    CHECK_ERROR_MSG("Kernel Error");
}

__global__
static void _calc_dx_m_du_lmk(float *dx_m_du, const float *u_d, const float *position_d, int num_points, C_ScanPointCloud scan) {
    float dr_du[27];
    calc_dr_du(dr_du, u_d);

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

            dx_m_du[3*3*k + 3*i + j] = -sum;
        }
    }
}

static void calc_de_du_lmk(float *de_du_d,
                           const float *u_d, const float *position_d, C_ScanPointCloud scan) {
    dim3 dimThread = dim3(static_cast<unsigned int>(3 * scan.numLmks), 3);
    _calc_dx_m_du_lmk<<<1, dimThread>>>(de_du_d, u_d, position_d, scan.numLmks, scan);
    CHECK_ERROR_MSG("Kernel Error");
    scale_array<<<1,scan.numLmks*3*3>>>(de_du_d, scan.numLmks*3*3, 1.0f/sqrtf(scan.numLmks));
    CHECK_ERROR_MSG("Kernel Error");
}

__global__
static void _calc_dx_da_lmk(float *dx_m_da, const float *u_d, int num_points, C_PcaDeformModel model, C_ScanPointCloud scan) {
    float r[9];
    calc_r_from_u(r, u_d);

    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = num_points * 3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = model.shapeRank;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j=y_start; j<y_size; j+=y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            float sum = 0.0f;

            sum += r[0*3 + i] * model.shapeDeformBasis_d[model.dim*j + 3*scan.validModelLmks_d[k] + 0];
            sum += r[1*3 + i] * model.shapeDeformBasis_d[model.dim*j + 3*scan.validModelLmks_d[k] + 1];
            sum += r[2*3 + i] * model.shapeDeformBasis_d[model.dim*j + 3*scan.validModelLmks_d[k] + 2];

            dx_m_da[3*model.shapeRank*k+model.shapeRank*i+j] = -sum;
        }
    }
}

static void calc_de_da_lmk(float *de_da_d,
                           const float *u_d, C_PcaDeformModel model, C_ScanPointCloud scan) {
    const int numThread = 32;
    const int xRequired = scan.numLmks * 3;
    const int yRequired = model.shapeRank;
    dim3 dimBlock((xRequired + numThread - 1) / numThread, (yRequired + numThread - 1) / numThread);
    dim3 dimThread(numThread, numThread);
    _calc_dx_da_lmk<<<dimBlock, dimThread>>>(de_da_d, u_d, scan.numLmks, model, scan);
    CHECK_ERROR_MSG("Kernel Error");

    int dimBlock2 = (xRequired*yRequired + 512 - 1) / 512;
    int dimThread2 = 512;
    scale_array<<<dimBlock2,dimThread2>>>(de_da_d, scan.numLmks*3*model.shapeRank, 1.0f/sqrtf(scan.numLmks));
    CHECK_ERROR_MSG("Kernel Error");
}

void calc_residual_lmk(float *residual_d, const float *position_d, C_ScanPointCloud scan) {
    calc_error_lmk(residual_d, position_d, scan);
    scale_array<<<1,scan.numLmks*3>>>(residual_d, scan.numLmks*3, 1.0f/sqrtf(scan.numLmks));
    CHECK_ERROR_MSG("Kernel Error");
}


void calc_derivatives_lmk(float *dres_dt_d, float *dres_du_d, float *dres_da1_d, float *dres_da2_d,
                          const float *u_d, const float *position_before_tarnsform_d,
                          C_PcaDeformModel model, C_ScanPointCloud scan) {
    calc_de_dt_lmk(dres_dt_d, scan.numLmks);
    calc_de_du_lmk(dres_du_d, u_d, position_before_tarnsform_d, scan);
    calc_de_da_lmk(dres_da1_d, u_d, model, scan);
    calc_de_da_lmk(dres_da2_d, u_d, model, scan);
}
