#include <math.h>
#include "util/cu_quaternion.h"

#define EPS 1e-10
#define SQ(X) (powf(X, 2.0f))
#define CU(X) (powf(X, 3.0f))

__device__
static float safe_norm(int dim, const float *v) {
    float norm2 = normf(dim, v);
    if (norm2 == 0) {
        // for avoiding zero-norm
        norm2 += EPS;
    }
    return norm2;
}

__device__
static void zero_elements(int dim, float *v) {
    for (int i=0; i<dim; i++) {
        v[i] = 0;
    }
}

__device__
static void scale_elements(float *out, const float scale, int dim, const float *v) {
    for (int i=0; i<dim; i++) {
        out[i] = v[i]*scale;
    }
}

__device__
void calc_r_from_u(float *r_d, const float *u_d) {
    float q[4];
    calc_q(q, u_d);
    calc_r_from_q(r_d, q);
}

__device__
void calc_r_from_q(float *r_d, const float *q_d) {
    const float *q = q_d;
    float r[9] = {
        SQ(q[0])+SQ(q[1])-SQ(q[2])-SQ(q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]-q[0]*q[2]),
        2*(q[1]*q[2]-q[0]*q[3]), SQ(q[0])-SQ(q[1])+SQ(q[2])-SQ(q[3]), 2*(q[2]*q[3]+q[0]*q[1]),
        2*(q[1]*q[3]+q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), SQ(q[0])-SQ(q[1])-SQ(q[2])+SQ(q[3])
    };
    for(int i=0; i<9; i++) {
        r_d[i] = r[i];
    }
}

__device__
void calc_dr_du(float *dr_du_d, const float *u_d) {
    //Convert u to q
    float q[4];
    calc_q(q, u_d);

    //Calculate dR(q(u))/dq(u)
    float dr_dq[4*9];
    calc_dr_dq(dr_dq, q);

    //Calculate dq(u)/du
    float dq_du[3*4];
    calc_dq_du(dq_du, u_d);

    for (int i=0; i<3; i++) {
        float *dr_dui = dr_du_d + i*9;
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

__device__
void calc_dr_dq(float *dr_dq_d, const float *q_d) {
    const float *q = q_d;
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
        dr_dq_d[0*9 + i] = dr_dq0[i];
        dr_dq_d[1*9 + i] = dr_dq1[i];
        dr_dq_d[2*9 + i] = dr_dq2[i];
        dr_dq_d[3*9 + i] = dr_dq3[i];
    }
}

__device__
void calc_q(float *q_d, const float *u_d) {
    float v = safe_norm(3, u_d);
    float u_normalizer = sinf(v / 2.0f) / v;
    q_d[0] = cosf(v / 2.0f);
    q_d[1] = u_normalizer * u_d[0];
    q_d[2] = u_normalizer * u_d[1];
    q_d[3] = u_normalizer * u_d[2];
}

__device__
void calc_dq_du(float *dq_du_d, const float *u_d) {
    const float *u = u_d;
    const float v = safe_norm(3, u_d);
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
        dq_du_d[0*4 + i] = dq_du0[i];
        dq_du_d[1*4 + i] = dq_du1[i];
        dq_du_d[2*4 + i] = dq_du2[i];
    }
}
