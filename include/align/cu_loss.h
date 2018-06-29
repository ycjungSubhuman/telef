#pragma once

#include <cuda_runtime_api.h>
#include "util/cu_quaternion.h"
#include "face/raw_model.h"

/**
 * Calculation of MSE losses and its derivatives
 */


/**
 * Calculate
 *
 *       N    3
 *      SUM  SUM  ( m_ik - x_m_ik )^2
 *       k    i
 *
 * Where
 *   m is the number of landmark points
 *   m_ik is k-th landmark point's i-th element
 *   x_m is the subset of mesh points that corresponds to landmark points
 *   x_m_ik is the k-th landmark point's i-th element
 *
 * @param mse_d                 calculated MSE. scalar
 * @param position_d            calculated mesh vertex positions
 * @param scan
 */
void calc_mse_lmk(float *mse_d, const float *position_d, C_PcaDeformModel model, C_ScanPointCloud scan);

/**
 * Calculate derivatives of landmark term
 *
 * Translation parameters:
 *
 *               N
 * de_dt_d[j] = SUM  -2 * ( m_jk - x_m_jk )
 *               k
 *
 * Rotation parameters
 *
 *               N   3
 * de_du_d[j] = SUM SUM -2 * ( m_ik - x_m_ik ) * (dr_duj_i `dot` x_m_k)
 *               k   i
 *
 * PCA coefficients
 *
 *               N   3
 * de_da_d[j] = SUM SUM -2 * ( m_ik - x_m_ik ) * (r_i `dot` v_j_k)
 *               k   i
 *
 * Where
 *   N is the number of landmark points
 *   m_jk is k-th landmark point's j-th element
 *   x_m is the subset of mesh points that corresponds to landmark points
 *   x_m_jk is the k-th landmark point's j-th element
 *   x_m_k is k-th point of x_m
 *   r_i is R(q(u))'s i-th row
 *   v_j_k is k-th point of j-th pca basis for deformation model
 *   dr_duj_i is derivative of rotation matrix wrt u_j
 *
 * @param de_dt_d               de_dt_d[j] = de_dtj. 3-element array
 * @param de_du_d               de_du_d[j] = de_duj. 3-element array
 * @param de_da_d               de_da_d[j] = de_daj. (rank of pca model)-element array
 * @param u_d                   rotation parameter. 3-element array. Axis-angle notation (see cu_quaternion.h)
 * @param position_d            calculated mesh vertex positions
 * @param scan
 */
void calc_derivatives_lmk(float *de_dt_d, float *de_du_d, float *de_da_d,
                          const float *u_d, const float *position_d, C_ScanPointCloud scan);

__device__
void calc_dll_du(float *dll_du_d, const float *u_d);

__device__
void calc_dll_dt(float *dll_du_d, const float *t_d);

