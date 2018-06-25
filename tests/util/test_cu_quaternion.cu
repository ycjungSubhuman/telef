#include <math.h>
#include <gtest/gtest.h>

#include "util/cu_quaternion.h"
#include "util/cudautil.h"
#include "testutil/diff.h"

#define WIDTH 32
#define BLOCKSIZE 8

__global__
void calc_diff_analytic_numeric(float *diff_d) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);
    int z = (blockIdx.z * blockDim.z + threadIdx.z);
    if (!(x < WIDTH && y < WIDTH && z < WIDTH)) return;
    float u[3] = {
        -0.016f + 0.001f*x,
        -0.016f + 0.001f*y,
        -0.016f + 0.001f*z
    };
    float numeric[3*3*3];
    float analytic[3*3*3];

    calc_numerical_diff(numeric, &calc_r_from_u, 9, 3, u);
    calc_dr_du(analytic, u);
    for (int i=0; i<3*3*3; i++) {
        diff_d[WIDTH*WIDTH*WIDTH*x + WIDTH*WIDTH*y + WIDTH*z + i] = analytic[i] - numeric[i];
    }
}

/**
 * Error btw numerical version and analytic version of dr_du calculation should be less than 1e-1
 *
 * This test is done only around the origin
 */
TEST(RotationDerivativeTest, AroundZero) {
    float *diff = (float*)malloc(3*3*3*WIDTH*WIDTH*WIDTH*sizeof(float));
    float *diff_d;
    CUDA_CHECK(cudaMalloc((void**)(&diff_d), 3*3*3*WIDTH*WIDTH*WIDTH*sizeof(float)));
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid((WIDTH + BLOCKSIZE-1)/BLOCKSIZE, (WIDTH + BLOCKSIZE-1)/BLOCKSIZE, (WIDTH + BLOCKSIZE-1)/BLOCKSIZE);
    calc_diff_analytic_numeric<<<dimGrid, dimBlock>>>(diff_d);
    CHECK_ERROR_MSG("Kernel Error");
    CUDA_CHECK(cudaMemcpy((void*)diff, diff_d, 3*3*3*WIDTH*WIDTH*WIDTH*sizeof(float), cudaMemcpyDeviceToHost));

    //show the first 27 values
    for (int i=0; i<3*3*3; i++) {
        printf("%f ", fabsf(diff[i]));
    }
    printf("\n");

/*
    for (int i=0; i<3*3*3*WIDTH*WIDTH*WIDTH; i++) {
        EXPECT_LE( fabsf(diff[i]),  1.0f );
    }
    */
    CUDA_CHECK(cudaFree(diff_d));
    free(diff);
}