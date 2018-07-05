#define _USE_MATH_DEFINES
#include <math.h>
#include <gtest/gtest.h>

#include "util/cudautil.h"
#include "../testutil/diff.h"

#define WIDTH 16
#define BLOCKSIZE 512

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
 * @param width                 parameter grid intersection count for each dimension
 */
__global__
void calc_diff_analytic_numeric(float *diff_d,
                                func *calc_analytic_diff,
                                func *calc_function_value,
                                float initial_value, float delta,
                                int param_dim, int val_dim,
                                const int width, bool first_zero) {
    int d = (blockIdx.x * blockDim.x + threadIdx.x);
    int *inds = (int*)malloc(param_dim*sizeof(int));
    float *x = (float*)malloc(param_dim*sizeof(float));

    // Setup index and function parameter
    for(int i=param_dim-1; i>=0; i--) {
        inds[i] = d % width;
        d /= width;
        if(first_zero) {
            x[i] = 0;
            first_zero = false;
            continue;
        }
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


__device__ func pcalc_dr_du = calc_dr_du;
__device__ func pcalc_r_from_u = calc_r_from_u;

void test_dr_du(float initial, float delta, bool first_zero=false) {
    float *diff = (float*)malloc(3*3*3*WIDTH*WIDTH*WIDTH*sizeof(float));
    float *diff_d;
    CUDA_CHECK(cudaMalloc((void**)(&diff_d), 3*3*3*WIDTH*WIDTH*WIDTH*sizeof(float)));
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((WIDTH*WIDTH*WIDTH + BLOCKSIZE-1)/BLOCKSIZE);

    // Copy Function Pointers
    func* calc_dr_du_f = (func*)malloc(sizeof(func));
    func* calc_r_from_u_f = (func*)malloc(sizeof(func));
    func *calc_dr_du_d, *calc_r_from_u_d;
    CUDA_CHECK(cudaMalloc((void**)&calc_dr_du_d, sizeof(func)));
    CUDA_CHECK(cudaMalloc((void**)&calc_r_from_u_d, sizeof(func)));
    CUDA_CHECK(cudaMemcpyFromSymbol(calc_dr_du_f, pcalc_dr_du, sizeof(func)));
    CUDA_CHECK(cudaMemcpyFromSymbol(calc_r_from_u_f, pcalc_r_from_u, sizeof(func)));
    CUDA_CHECK(cudaMemcpy(calc_dr_du_d, calc_dr_du_f, sizeof(func), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(calc_r_from_u_d, calc_r_from_u_f, sizeof(func), cudaMemcpyHostToDevice));

    // Run difference calculation
    calc_diff_analytic_numeric<<<dimGrid, dimBlock>>>(diff_d, calc_dr_du_d, calc_r_from_u_d, initial, delta, 3, 9, WIDTH, first_zero);
    CHECK_ERROR_MSG("Kernel Error");

    CUDA_CHECK(cudaMemcpy((void*)diff, diff_d, 3*3*3*WIDTH*WIDTH*WIDTH*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i=0; i<3*3*3*WIDTH*WIDTH*WIDTH; i++) {
        EXPECT_LE( fabsf(diff[i]),  0.01f );
    }

    //show the 27 values at zero
    printf("Differences(27 samples): ");
    for (int i=WIDTH*WIDTH*WIDTH; i<WIDTH*WIDTH*WIDTH+3*3*3; i++) {
        printf("%f ", fabsf(diff[i]));
    }
    printf("\n");

    CUDA_CHECK(cudaFree(diff_d));
    CUDA_CHECK(cudaFree(calc_dr_du_d));
    CUDA_CHECK(cudaFree(calc_r_from_u_d));
    free(diff);
    free(calc_dr_du_f);
    free(calc_r_from_u_f);
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (0, 0, 0)
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundZero) {
    test_dr_du(-0.08f, 0.01f);
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (1, 1, 1)
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundOne) {
    test_dr_du(0.92f, 0.01f);
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (-1, -1, -1)
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundMinusOne) {
    test_dr_du(-1.08f, 0.01f);
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (1, 1, 1)*(PI/sqrt(3))
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundPI) {
    test_dr_du(0.92f * (M_PI / sqrtf(3.0f)), 0.01f * (M_PI / sqrtf(3.0f)));
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (-1, -1, -1)*(PI/sqrt(3.0f))
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundMinusPI) {
    test_dr_du(-1.08f * (M_PI / sqrtf(3.0f)), 0.01f * (M_PI / sqrtf(3.0f)));
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (0, -1, -1)*(PI/sqrt(3.0f))
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundMinusPIZeroU0) {
    test_dr_du(-1.08f * (M_PI / sqrtf(3.0f)), 0.01f * (M_PI / sqrtf(3.0f)), true);
}
