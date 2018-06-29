#include <math.h>
#include "util/cu_quaternion.h"

#define EPS 1e-6
#define SQ(X) (powf(X, 2.0f))
#define CU(X) (powf(X, 3.0f))

__device__ __host__
static float safe_3_elem_norm(int dim, const float *v) {
#ifdef __CUDA_ARCH__
    float norm2 = normf(dim, v);
#else
    float norm2 = sqrtf(powf(v[0], 2.0f) + powf(v[1], 2.0f) + powf(v[2], 2.0f));
#endif
    if (norm2 == 0) {
        // for avoiding zero-norm
        norm2 += EPS;
    }
    return norm2;
}

__device__ __host__
static void zero_elements(int dim, float *v) {
    for (int i=0; i<dim; i++) {
        v[i] = 0;
    }
}

__device__ __host__
static void scale_elements(float *out, const float scale, int dim, const float *v) {
    for (int i=0; i<dim; i++) {
        out[i] = v[i]*scale;
    }
}

__device__ __host__
void calc_r_from_u(float *r, const float *u) {
    float q[4];
    calc_q(q, u);
    calc_r_from_q(r, q);
}

__device__ __host__
void calc_r_from_q(float *r, const float *q) {
    float _r[9] = {
        SQ(q[0])+SQ(q[1])-SQ(q[2])-SQ(q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]-q[0]*q[2]),
        2*(q[1]*q[2]-q[0]*q[3]), SQ(q[0])-SQ(q[1])+SQ(q[2])-SQ(q[3]), 2*(q[2]*q[3]+q[0]*q[1]),
        2*(q[1]*q[3]+q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), SQ(q[0])-SQ(q[1])-SQ(q[2])+SQ(q[3])
    };
    for(int i=0; i<9; i++) {
        r[i] = _r[i];
    }
}

__device__ __host__
void calc_dr_du(float *dr_du, const float *u) {
    //Convert u to q
    float q[4];
    calc_q(q, u);

    //Calculate dR(q(u))/dq(u)
    float dr_dq[4*9];
    calc_dr_dq(dr_dq, q);

    //Calculate dq(u)/du
    float dq_du[3*4];
    calc_dq_du(dq_du, u);

    for (int i=0; i<3; i++) {
        float *dr_dui = dr_du + i*9;
        zero_elements(9, dr_dui); // initialization

        for (int j=0; j<4; j++) {
            float dr_dui_j[9];
            scale_elements(dr_dui_j, dq_du[4*i+j], 9, (dr_dq + 9*j));
            for (int k=0; k<9; k++) {
                dr_dui[k] += dr_dui_j[k];
            }
        }
    }
}

__device__ __host__
void calc_dr_dq(float *dr_dq, const float *q) {
    const float dr_dq0[9] = {
        2*q[0], 2*q[3], -2*q[2],
        -2*q[3], 2*q[0], 2*q[1],
        2*q[2], -2*q[1], 2*q[0]
    };
    const float dr_dq1[9] = {
        2*q[1], 2*q[2], 2*q[3],
        2*q[2], -2*q[1], 2*q[0],
        2*q[3], -2*q[0], -2*q[1]
    };
    const float dr_dq2[9] = {
        -2*q[2], 2*q[1], -2*q[0],
        2*q[1], 2*q[2], 2*q[3],
        2*q[0], 2*q[3], -2*q[2]
    };
    const float dr_dq3[9] = {
        -2*q[3], 2*q[0], 2*q[1],
        -2*q[0], -2*q[3], 2*q[2],
        2*q[1], 2*q[2], 2*q[3]
    };

    for (int i=0; i<9; i++) {
        dr_dq[0*9 + i] = dr_dq0[i];
        dr_dq[1*9 + i] = dr_dq1[i];
        dr_dq[2*9 + i] = dr_dq2[i];
        dr_dq[3*9 + i] = dr_dq3[i];
    }
}

__device__ __host__
void calc_q(float *q, const float *u) {
    float v = safe_3_elem_norm(3, u);
    float u_normalizer = sinf(v / 2.0f) / v;
    q[0] = cosf(v / 2.0f);
    q[1] = u_normalizer * u[0];
    q[2] = u_normalizer * u[1];
    q[3] = u_normalizer * u[2];
}

__device__ __host__
void calc_dq_du(float *dq_du, const float *u) {
    const float v = safe_3_elem_norm(3, u);
    const float s = sinf(v / 2.0f);
    const float c = cosf(v / 2.0f);
    const float c_2v2 = c/(2*SQ(v));
    const float s_v = s/v;
    const float s_v3 = s/CU(v);
    const float c_m_s = c_2v2 - s_v3;

    const float dq_du0[4] = {
        -u[0]*s_v/2.0f,
        s_v + SQ(u[0])*c_2v2 - SQ(u[0])*s_v3,
        u[0]*u[1]*c_m_s,
        u[0]*u[2]*c_m_s
    };
    const float dq_du1[4] = {
        -u[1]*s_v/2.0f,
        u[0]*u[1]*c_m_s,
        s_v + SQ(u[1])*c_2v2 - SQ(u[1])*s_v3,
        u[1]*u[2]*c_m_s
    };
    const float dq_du2[4] = {
        -u[2]*s_v/2.0f,
        u[0]*u[2]*c_m_s,
        u[1]*u[2]*c_m_s,
        s_v + SQ(u[2])*c_2v2 - SQ(u[2])*s_v3
    };

    for (int i=0; i<4; i++) {
        dq_du[0*4 + i] = dq_du0[i];
        dq_du[1*4 + i] = dq_du1[i];
        dq_du[2*4 + i] = dq_du2[i];
    }
}
