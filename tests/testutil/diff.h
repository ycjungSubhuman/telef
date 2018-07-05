#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "util/cu_quaternion.h"

typedef void(*func)(float*, const float*);

//TODO : The selection of h only works for specific function and specific input. Make it adaptive.
/**
 * Calculate numerical derivative value on x0
 *
 * F'(x) = (F(x+h) - F(x-h)) /  (2h)
 *
 * for a small h
 *
 * @param result                        the result array of length val_dim
 * @param calc_func                     function pointer for evaluating function value from parameter
 * @param h                             small value h
 *                                      the optimal h value are dependent on the shape of function.
 * @param val_dim                       the dimension of function value
 * @param param_dim                     the dimension of parameter
 * @param x0                            the point we want to evaluate numerical diff on
 */
__device__ __host__
void calc_numerical_diff(float *result, func *calc_func,
                         const float h, int val_dim, int param_dim, const float *x0) {
    float *x1 = (float*)malloc(param_dim*sizeof(float));
    float *x2 = (float*)malloc(param_dim*sizeof(float));
    float *f1 = (float*)malloc(val_dim*sizeof(float));
    float *f2= (float*)malloc(val_dim*sizeof(float));

    for (int i=0; i<param_dim; i++) {
        for (int j=0; j<param_dim; j++) {
            if(i==j) {
                x1[j] = x0[j]-h;
            }
            else {
                x1[j] = x0[j];
            }
        }

        for (int j=0; j<param_dim; j++) {
            if(i==j) {
                x2[j] = x0[j]+h;
            }
            else {
                x2[j] = x0[j];
            }
        }

        (*calc_func)(f2, x2);
        (*calc_func)(f1, x1);

        for (int j=0; j<val_dim; j++) {
            result[val_dim*i + j] = (double)(f2[j] - f1[j]) / (2.0f * h);
        }
    }

    free(x1);
    free(x2);
    free(f1);
    free(f2);
}
