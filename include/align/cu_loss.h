#pragma once

#include <cuda_runtime_api.h>
#include "util/cu_quaternion.h"
#include "face/raw_model.h"

/**
 * Calculation of residuals and its derivatives
 */

typedef struct PointPair{
    float *mesh_position_d;
    float *mesh_positoin_before_transform_d;
    float *ref_position_d;
    int *mesh_corr_inds_d;
    int *ref_corr_inds_d;
    int point_count;
} PointPair;

/**
 * Calculate
 *
 *                          1
 * residual_d[3*k + i] = -------  ( m_ik - x_m_ik )
 *                       sqrt(N)
 *
 * Where
 *   m is the number of points
 *   m_ik is k-th correspondence point's i-th element
 *   x_m is the subset of mesh points that corresponds to points
 *   x_m_ik is the k-th landmark point's i-th element
 */
void calc_residual_point_pair(float *residual_d, PointPair point_pair);

/**
 * Calculate derivatives of least squares
 *
 * Translation parameters:
 *
 *                                             1
 * dres_dt_d[3*3*k+3*i+j] = if i==j then (- -------) else 0
 *                                          sqrt(N)
 *
 *
 * Rotation parameters
 *
 *                               1
 * dres_du_d[3*3*k+3*i+j] = - ------- (dr_duj_i `dot` x_m_k)
 *                            sqrt(N)
 *
 * PCA coefficients
 *
 *                               1
 * dres_da_d[3*R*k+R*i+j] = - ------- (r_i `dot` v_j_k)
 *                            sqrt(N)
 *
 * Where
 *   N is the number of landmark points
 *   R is the number of PCA coefficients
 *   m_jk is k-th landmark point's j-th element
 *   x_m is the subset of mesh points that corresponds to landmark points
 *   x_m_jk is the k-th landmark point's j-th element
 *   x_m_k is k-th point of x_m
 *   r_i is R(q(u))'s i-th row
 *   v_j_k is k-th point of j-th pca basis for deformation model
 *   dr_duj_i is derivative of rotation matrix wrt u_j
 *
 * @param dres_dt_d               de_dt_d[j] = de_dtj. ((number of landmarks)x3x3)-element array
 * @param dres_du_d               de_du_d[j] = de_duj. ((number of landmarks)x3x3)-element array
 * @param dres_da1_d              de_da1_d[j] = de_da1j. ((number of landmarks)x3x(rank of pca model))-element array
 * @param dres_da2_d              de_da2_d[j] = de_da2j. ((number of landmarks)x3x(rank of pca model))-element array
 * @param u_d                   rotation parameter. 3-element array. Axis-angle notation (see cu_quaternion.h)
 * @param model
 * @param point_pair
 */
void calc_derivatives_point_pair(float *dres_dt_d, float *dres_du_d, float *dres_da1_d, float *dres_da2_d,
                                 const float *u_d, C_PcaDeformModel model, PointPair point_pair);
