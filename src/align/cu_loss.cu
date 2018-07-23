#include <device_launch_parameters.h>
#include <math.h>
#include "align/cu_loss.h"
#include "util/cu_quaternion.h"
#include "util/cu_array.h"
#include "util/cudautil.h"

namespace {
    const int NUM_THREAD = 512;
    const int DIM_X_THREAD = 16;
    const int DIM_Y_THREAD = 16;
}

/**
 * Calculate error of each element of each point, resulting in 3xN array
 *
 * error = (scan - position)
 */
__global__
static void _calc_error_lmk(float *error_d, PointPair point_pair) {
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = point_pair.point_count*3;
    const int step = blockDim.x * gridDim.x;
    for(int ind=start; ind<size; ind+=step) {
        const int k = ind / 3;
        const int ref_ind = point_pair.ref_corr_inds_d[k];
        const int mesh_ind = point_pair.mesh_corr_inds_d[k];
        const int i = ind % 3;
        const float m_ik = point_pair.ref_position_d[3*ref_ind + i];
        const float x_m_ik = point_pair.mesh_position_d[3*mesh_ind + i];

        const float error = m_ik - x_m_ik;
        error_d[ind] = error;
    }
}

static void calc_error_lmk(float *error_d, PointPair point_pair) {
    // This is used as cache before sum reduction
    const int threadRequired = point_pair.point_count*3;
    _calc_error_lmk<<<GET_DIM_GRID(threadRequired, NUM_THREAD), NUM_THREAD>>>(error_d, point_pair);
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

static void calc_de_dt_lmk(float *de_dt_d, int num_points, const float weight) {
    const int xRequired = 3*num_points;
    const int yRequired = 3;
    dim3 dimGrid(GET_DIM_GRID(xRequired, DIM_X_THREAD), GET_DIM_GRID(yRequired, DIM_Y_THREAD));
    dim3 dimBlock(DIM_X_THREAD, DIM_Y_THREAD);
    _calc_dx_m_dt_lmk<<<dimGrid, dimBlock>>>(de_dt_d, num_points);
    CHECK_ERROR_MSG("Kernel Error");
    scale_array<<<GET_DIM_GRID(num_points*3*3, NUM_THREAD), NUM_THREAD>>>
                    (de_dt_d, num_points*3*3, weight*1.0f/sqrtf(num_points));
    CHECK_ERROR_MSG("Kernel Error");
}

__global__
static void _calc_dx_m_du_lmk(float *dx_m_du, const float *u_d, PointPair point_pair) {
    float dr_du[27];
    calc_dr_du(dr_du, u_d);

    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = point_pair.point_count * 3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = 3;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j=y_start; j<y_size; j+=y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            const int mesh_idx = point_pair.mesh_corr_inds_d[k];
            float sum = 0.0f;

            sum += dr_du[9*j + 3*0 + i] * point_pair.mesh_positoin_before_transform_d[3*mesh_idx + 0];
            sum += dr_du[9*j + 3*1 + i] * point_pair.mesh_positoin_before_transform_d[3*mesh_idx + 1];
            sum += dr_du[9*j + 3*2 + i] * point_pair.mesh_positoin_before_transform_d[3*mesh_idx + 2];

            dx_m_du[3*3*k + 3*i + j] = -sum;
        }
    }
}

static void calc_de_du_lmk(float *de_du_d, const float *u_d, PointPair point_pair, const float weight) {
    const int xRequired = 3*point_pair.point_count;
    const int yRequired = 3;
    dim3 dimGrid(GET_DIM_GRID(xRequired, DIM_X_THREAD), GET_DIM_GRID(yRequired, DIM_Y_THREAD));
    dim3 dimBlock(DIM_X_THREAD, DIM_Y_THREAD);
    _calc_dx_m_du_lmk<<<dimGrid, dimBlock>>>(de_du_d, u_d, point_pair);
    CHECK_ERROR_MSG("Kernel Error");
    scale_array<<<GET_DIM_GRID(3*3*point_pair.point_count, NUM_THREAD),NUM_THREAD>>>
                   (de_du_d, point_pair.point_count*3*3, weight*1.0f/sqrtf(point_pair.point_count));
    CHECK_ERROR_MSG("Kernel Error");
}

__global__
static void _calc_dx_da_lmk(float *dx_m_da, const float *u_d, int rank, int dim, const float *basis_d, PointPair point_pair) {
    float r[9];
    calc_r_from_u(r, u_d);

    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = point_pair.point_count*3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = rank;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j=y_start; j<y_size; j+=y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            const int mesh_ind = point_pair.mesh_corr_inds_d[k];
            float sum = 0.0f;

            sum += r[0*3 + i] * basis_d[dim*j + 3*mesh_ind + 0];
            sum += r[1*3 + i] * basis_d[dim*j + 3*mesh_ind + 1];
            sum += r[2*3 + i] * basis_d[dim*j + 3*mesh_ind + 2];

            dx_m_da[3*rank*k+rank*i+j] = -sum;
        }
    }
}

static void calc_de_da_lmk(float *de_da_d,
                           const float *u_d, int rank, int dim, const float *basis_d, PointPair point_pair, const float weight) {
    const int xRequired = point_pair.point_count* 3;
    const int yRequired = rank;
    dim3 dimGrid(GET_DIM_GRID(xRequired, DIM_X_THREAD), GET_DIM_GRID(yRequired, DIM_Y_THREAD));
    dim3 dimBlock(DIM_X_THREAD, DIM_Y_THREAD);
    _calc_dx_da_lmk<<<dimGrid, dimBlock>>>(de_da_d, u_d, rank, dim, basis_d, point_pair);
    CHECK_ERROR_MSG("Kernel Error");

    scale_array<<<GET_DIM_GRID(xRequired*yRequired, NUM_THREAD), NUM_THREAD>>>
                 (de_da_d, point_pair.point_count*3*rank, weight*1.0f/sqrtf(point_pair.point_count));
    CHECK_ERROR_MSG("Kernel Error");
}

void calc_residual_point_pair(float *residual_d, PointPair point_pair, const float weight) {
    calc_error_lmk(residual_d, point_pair);
    scale_array<<<GET_DIM_GRID(point_pair.point_count*3, NUM_THREAD), NUM_THREAD>>>
                      (residual_d, point_pair.point_count*3, weight*1.0f/sqrtf(point_pair.point_count));
    CHECK_ERROR_MSG("Kernel Error");
}


void calc_derivatives_point_pair(float *dres_dt_d, float *dres_du_d, float *dres_da1_d, float *dres_da2_d,
                                 const float *u_d, C_PcaDeformModel model, PointPair point_pair, const float weight) {
    calc_de_dt_lmk(dres_dt_d, point_pair.point_count, weight);
    calc_de_du_lmk(dres_du_d, u_d, point_pair, weight);
    calc_de_da_lmk(dres_da1_d, u_d, model.shapeRank, model.dim, model.shapeDeformBasis_d, point_pair, weight);
    calc_de_da_lmk(dres_da2_d, u_d, model.expressionRank, model.dim, model.expressionDeformBasis_d, point_pair, weight);
}
