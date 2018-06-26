#pragma once
#include <cstdio>
#include <cuda_runtime_api.h>
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
__device__
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

/**
 * Calculate difference between analytic differentiation and numeric differentiation
 *
 * Tests on various parameters starting from (initial_value, initial_value ...)
 *
 * example ) 2D parameter case
 *
 *  initial_value
 *  |
 *  V
 *  *---delta---*---delta---*---delta---* ...
 *  |           |           |
 *  delta     delta       delta
 *  |           |           |
 *  *---delta---*---delta---*---delta---* ...
 *  ...
 *
 *  (Total number of grid intersections '*') = width^param_dim
 *
 * @param diff_d                output matrix. width^param_dim * val_dim*param_dim
 *                              stores element wise differnce btw analytic and numeric derivatives
 * @param calc_analytic_diff    function pointer for calculating analytic differentiation
 * @param calc_function_value   function pointer for calculating function value
 * @param initial_value         initial parameter value
 * @param delta                 parameter perturbation amount
 * @param param_dim             parameter dimension
 * @param val_dim               functino value dimension(number of elements calculated from calc_function_value)
 * @param width                 parameter grid count for each dimension
 */
__global__
void calc_diff_analytic_numeric(float *diff_d,
                                func *calc_analytic_diff,
                                func *calc_function_value,
                                float initial_value, float delta,
                                int param_dim, int val_dim,
                                const int width) {
    int d = (blockIdx.x * blockDim.x + threadIdx.x);
    int *inds = (int*)malloc(param_dim*sizeof(int));
    float *x = (float*)malloc(param_dim*sizeof(float));

    // Setup index and function parameter
    for(int i=param_dim-1; i>=0; i--) {
        inds[i] = d % width;
        d /= width;
        x[i] = initial_value + inds[i]*delta;
    }

    float *numeric = (float*)malloc(param_dim*val_dim*sizeof(float));
    float *analytic = (float*)malloc(param_dim*val_dim*sizeof(float));

    calc_numerical_diff(numeric, calc_function_value, 1e-4, val_dim, param_dim, x);
    (*calc_analytic_diff)(analytic, x);

    int diff_index=0;
    for (int i=0; i<param_dim; i++) {
        diff_index = (diff_index + inds[i])*width;
    }

    for (int i=0; i<3*3*3; i++) {
        diff_d[diff_index + i] = analytic[i] - numeric[i];
    }

    if (diff_index==0) {
        printf("Analytic(27 samples): ");
        for (int i=0; i<3*3*3; i++) {
            printf("%f/", analytic[i]);
        }
        printf("\n");

        printf("Numeric(27 samples): ");
        for (int i=0; i<3*3*3; i++) {
            printf("%f/", numeric[i]);
        }
        printf("\n");
    }

    free(inds);
    free(x);
    free(numeric);
    free(analytic);
}

