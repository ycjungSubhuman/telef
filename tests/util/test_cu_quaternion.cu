#include "util/cu_quaternion.h"
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>

#define WIDTH 32
#define BLOCKSIZE 8

#define CUDA_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)

inline void __checkErrorFunc(cudaError_t errarg, const char* file,
                 const int line)
{
    if(errarg) {
    fprintf(stderr, "Error at %s(%i)\n", file, line);
    exit(EXIT_FAILURE);
    }
}


inline void __checkErrMsgFunc(const char* errstr, const char* file,
                  const int line)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
    fprintf(stderr, "Error: %s at %s(%i): %s\n",
        errstr, file, line, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
}

__device__
void calc_numerical_diff(float *result, void (*calc_func)(float*, const float*), int val_dim, int param_dim, const float *x0) {
    float *x1 = (float*)malloc(param_dim*sizeof(float));
    float *x2 = (float*)malloc(param_dim*sizeof(float));
    float *f0 = (float*)malloc(val_dim*sizeof(float));
    float *f = (float*)malloc(val_dim*sizeof(float));

    const double eps = 1e-4;

    for (int i=0; i<param_dim; i++) {
        for (int j=0; j<param_dim; j++) {
            if(i==j) {
                x1[i] = x0[i]+eps;
            }
            else {
                x1[i] = x0[i];
            }
        }

        for (int j=0; j<param_dim; j++) {
            if(i==j) {
                x2[i] = x0[i]-eps;
            }
            else {
                x2[i] = x0[i];
            }
        }

        calc_func(f0, x2);
        calc_func(f, x1);

        for (int j=0; j<val_dim; j++) {
            result[val_dim*i + j] = (f[j] - f0[j]) / (2*eps);
        }
    }

    free(x1);
    free(x2);
    free(f0);
    free(f);
}

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
 * Error btw numerical version and analytic version of dr_du calculation should be less than 1e-4
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
    CUDA_CHECK(cudaMemcpy((void*)diff, diff_d, 3*3*3*sizeof(float), cudaMemcpyDeviceToHost));

    //show the first 27 values
    for (int i=0; i<3*3*3; i++) {
        printf("%f ", fabsf(diff[i]));
    }
    printf("\n");

    for (int i=0; i<3*3*3*WIDTH*WIDTH*WIDTH; i++) {
        EXPECT_LE( fabsf(diff[i]),  1.0f );
    }
    CUDA_CHECK(cudaFree(diff_d));
    free(diff);
}