#define _USE_MATH_DEFINES
#include <math.h>
#include <gtest/gtest.h>

#include "util/cudautil.h"
#include "../testutil/diff.h"

#define WIDTH 16
#define BLOCKSIZE 512

__device__ func pcalc_dr_du = calc_dr_du;
__device__ func pcalc_r_from_u = calc_r_from_u;

void test_dr_dq(float initial, float delta) {
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
    calc_diff_analytic_numeric<<<dimGrid, dimBlock>>>(diff_d, calc_dr_du_d, calc_r_from_u_d, initial, delta, 3, 9, WIDTH);
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
    test_dr_dq(-0.08f, 0.01f);
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (1, 1, 1)
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundOne) {
    test_dr_dq(0.92f, 0.01f);
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (-1, -1, -1)
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundMinusOne) {
    test_dr_dq(-1.08f, 0.01f);
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (1, 1, 1)*(PI/sqrt(3))
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundPI) {
    test_dr_dq(0.92f*(M_PI/sqrtf(3.0f)), 0.01f*(M_PI/sqrtf(3.0f)));
}

/**
 * Error btw numerical version and analytic version of dr_du calculation around
 * (-1, -1, -1)*(PI/sqrt(3.0f))
 * should be less than 1e-2
 */
TEST(RotationDerivativeTest, AroundMinusPI) {
    test_dr_dq(-1.08f*(M_PI/sqrtf(3.0f)), 0.01f*(M_PI/sqrtf(3.0f)));
}

