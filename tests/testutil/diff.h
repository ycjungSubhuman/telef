#pragma once
#include <cstdio>
#include "util/cu_quaternion.h"

//TODO : This only works for specific function and specific input
__device__
void calc_numerical_diff(float *result, void (*calc_func)(float*, const float*), int val_dim, int param_dim, const float *x0) {
    float *x1 = (float*)malloc(param_dim*sizeof(float));
    float *x2 = (float*)malloc(param_dim*sizeof(float));
    float *f1 = (float*)malloc(val_dim*sizeof(float));
    float *f2= (float*)malloc(val_dim*sizeof(float));

    const float h = 1e-5;

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

        calc_func(f2, x2);
        calc_func(f1, x1);

        for (int j=0; j<val_dim; j++) {
            result[val_dim*i + j] = (double)(f2[j] - f1[j]) / (2.0f * h);
        }
    }

    free(x1);
    free(x2);
    free(f1);
    free(f2);
}

