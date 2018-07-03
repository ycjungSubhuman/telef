#define _USE_MATH_DEFINES
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "util/cudautil.h"
#include "../testutil/diff.h"

#include "face/cu_model_kernel.h"

using namespace testing;

#define BLOCKSIZE 128

//class NonRigidPCATest : public ::testing::Test {
//
//protected:
//
//    /**
//     * Host Vars
//     */
//    const float input_1[] = {0.416536, 0.312364, 0.033625};
//    const float input_2[] = {1.0, 1.0, 1.0,
//                             0.3, 0.5, -0.2};
//    const float translationMat[] = {0.942222, -0.0436802, -0.0112499, -0.0541017,
//                                    -0.0439274, -0.942033, -0.021433, -0.0656259,
//                                    -0.0102423, 0.0219324, -0.94299, 0.913377,
//                                    0, 0, 0, 1};
//
//
//    /**
//     * Device Vars
//     */
//    float *input_1_d;
//
//
//    virtual void SetUp() {
//        cudaMalloc(&input_1_d, 3 * sizeof(float));
//        cudaMemcpy(input_1_d, input_1, 3 * sizeof(float), cudaMemcpyHostToDevice);
//    }
//
//    virtual void TearDown() {
//        cudaFree(input_1_d);
//    }
//};

TEST(NonRigidPCATest, HomogeneousPoints) {
    dim3 grid = ((1+BLOCKSIZE-1)/BLOCKSIZE);
    dim3 block = BLOCKSIZE;

    const float input[] = {1.0, 1.0, 1.0};
    const float expect[] = {1.0, 1.0, 1.0, 1.0};

    float output[4];
    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, 3 * sizeof(float));
    cudaMalloc((void**)&output_d, 4 * sizeof(float));

    cudaMemcpy(input_d, input, 3 * sizeof(float), cudaMemcpyHostToDevice);

    _homogeneousPositions<<<grid,block>>>(output_d, input_d, 1);

    cudaMemcpy(output, output_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> output_vector(output, output + GTEST_ARRAY_SIZE_(output));
    EXPECT_THAT(output_vector, ElementsAreArray(expect));
}

TEST(NonRigidPCATest, HnormalizedPoints) {
    dim3 grid = ((1+BLOCKSIZE-1)/BLOCKSIZE);
    dim3 block = BLOCKSIZE;

    const float input[] = {2.0, 1.0, 4.0, 2.0};
    const float expect[] = {1.0, 0.5, 2.0};

    float output[3];
    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, 4 * sizeof(float));
    cudaMalloc((void**)&output_d, 3 * sizeof(float));

    cudaMemcpy(input_d, input, 4 * sizeof(float), cudaMemcpyHostToDevice);

    _hnormalizedPositions<<<grid,block>>>(output_d, input_d, 1);

    cudaMemcpy(output, output_d, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> output_vector(output, output + GTEST_ARRAY_SIZE_(output));
    EXPECT_THAT(output_vector, ElementsAreArray(expect));
}

TEST(NonRigidPCATest, CublasMatrixMultipy) {
    const float translationMat[] = {1.0,1.0,2.0,2.0};
    const float input[] = {1.0, 2.0};
    const float expect[] = {5.0, 5.0};

    float output[2];
    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, 2 * sizeof(float));
    cudaMalloc((void**)&output_d, 2 * sizeof(float));

    cudaMemcpy(input_d, input, 2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMatMul(output_d, translationMat, 2, 2, input_d, 2, 1);

    cudaMemcpy(output, output_d, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_THAT(output,ElementsAreArray(expect));
}

TEST(NonRigidPCATest, CublasMatrixMultipy2) {
    /**
     * Col-major order
     *
     * const float translationMat[] = {a,  e,  h,  0,
     *                                 b,  f,  i,  0,
     *                                 c,  g,  j,  0,
     *                                 tx, ty, tz, 1};
     */
    const float translationMat[] = {0.5, -0.5, -0.5, 0,
                                    -0.5, -0.5, 0.5, 0,
                                    -0.5, -0.5, -0.5, 0,
                                    -0.5, -0.5, 0.5, 1};

    const float input[] = {1.0, 1.0, 1.0, 1.0,
                           2.0, 2.0, 2.0, 2.0};
    const float expect[] = {-1, -2, 0, 1, -2, -4, 0, 2};

    float output[4*2];
    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, 4*2 * sizeof(float));
    cudaMalloc((void**)&output_d, 4*2 * sizeof(float));

    cudaMemcpy(input_d, input, 4*2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMatMul(output_d, translationMat, 4, 4, input_d, 4, 2);

    cudaMemcpy(output, output_d, 4*2 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_THAT(output,ElementsAreArray(expect));
}

TEST(NonRigidPCATest, RigidAlign) {
    /**
     * Col-major order
     *
     * const float translationMat[] = {a,  e,  h,  0,
     *                                 b,  f,  i,  0,
     *                                 c,  g,  j,  0,
     *                                 tx, ty, tz, 1};
     */
    const float translationMat[] = {0.5, -0.5, -0.5, 0,
                                    -0.5, -0.5, 0.5, 0,
                                    -0.5, -0.5, -0.5, 0,
                                    -0.5, -0.5, 0.5, 1};
    const float input[] = {1.0, 1.0, 1.0};
    const float expect[] = {-1, -2, 0};

    float output[3];
    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, 3 * sizeof(float));
    cudaMalloc((void**)&output_d, 3 * sizeof(float));

    cudaMemcpy(input_d, input, 3 * sizeof(float), cudaMemcpyHostToDevice);

    applyRigidAlignment(output_d, input_d, translationMat, 1);

    cudaMemcpy(output, output_d, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_THAT(output,ElementsAreArray(expect));
}

TEST(NonRigidPCATest, RigidAlign2) {
    /**
     * Col-major order
     *
     * const float translationMat[] = {a,  e,  h,  0,
     *                                 b,  f,  i,  0,
     *                                 c,  g,  j,  0,
     *                                 tx, ty, tz, 1};
     */
    const float translationMat[] = {0.942222, -0.0439274, -0.0102423,0,
                                    -0.0436802, -0.942033, 0.0219324,0,
                                    -0.0112499, -0.021433, -0.94299,0,
                                    -0.0541017, -0.0656259, 0.913377,1};
    const float input[] = {0.416536, 0.312364, 0.033625};
    const float expect[] = {0.324345, -0.378901, 0.884254};

    float output[3];
    float *input_d, *output_d;
    cudaMalloc((void**)&input_d, 3 * sizeof(float));
    cudaMalloc((void**)&output_d, 3 * sizeof(float));

    cudaMemcpy(input_d, input, 3 * sizeof(float), cudaMemcpyHostToDevice);

    applyRigidAlignment(output_d, input_d, translationMat, 1);

    cudaMemcpy(output, output_d, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    float ferr = 1e-5;
    EXPECT_THAT(output,
                Pointwise(FloatNear(ferr), expect));
}

TEST(NonRigidPCATest, ResidualAndJacobians_1) {
    int numLmks = 1;
    dim3 lmkThrds(BLOCKSIZE);
    dim3 lmkBlocks((numLmks + BLOCKSIZE - 1) / BLOCKSIZE);

    int npoints = 1;
    const float position[] = {1, 1, 1};
    const int lmks[] = {0};
    const float deformB[] = {1,1,1, 1,1,1, 1,1,1};
    int deformB_row = 3;
    int deformB_col = 3;
    int nScanPnts = 1;
    const float scanPoints[] = {2, 2, 2};
    const int scanLmks[] = {0};
    const bool isJacobianRequired = true;

    const float expect_res = 3.0;
    const float expect_jacobi[] = {6.0, 6.0, 6.0};

    float res;
    float jacobi[3];
    float *res_d, *jacobi_d;
    cudaMalloc((void**)&res_d, sizeof(float));
    cudaMalloc((void**)&jacobi_d, deformB_col*sizeof(float));

    float *position_d, *deformB_d, *scanPoints_d;

    CUDA_CHECK(cudaMalloc((void**)&position_d, 3 * npoints * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&deformB_d, deformB_row * deformB_col * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&scanPoints_d, 3 * nScanPnts * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(position_d, position, 3 * npoints * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deformB_d, deformB, deformB_row * deformB_col * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scanPoints_d, scanPoints, 3 * nScanPnts * sizeof(float), cudaMemcpyHostToDevice));

    int *lmks_d, *scanLmks_d;
    CUDA_CHECK(cudaMalloc((void**)&lmks_d, numLmks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&scanLmks_d, numLmks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(lmks_d, lmks, numLmks * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scanLmks_d, scanLmks, numLmks * sizeof(float), cudaMemcpyHostToDevice));

    _calculateLandmarkLoss << < lmkBlocks, lmkThrds >> >(res_d, jacobi_d, position_d,
            deformB_d, deformB_row, deformB_col, lmks_d, scanPoints_d, scanLmks_d, numLmks, isJacobianRequired);
    CHECK_ERROR_MSG("Kernel Error");

    CUDA_CHECK(cudaMemcpy(&res, res_d, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(jacobi, jacobi_d, deformB_col*sizeof(float), cudaMemcpyDeviceToHost));

//    printf("Resudual: %.6f\n", res);
//    printf("Jacobians: [0]=%.6f [1]=%.6f [2]=%.6f\n", jacobi[0], jacobi[1], jacobi[2]);

    float ferr = 1e-5;
    EXPECT_NEAR(res, expect_res, ferr);
    EXPECT_THAT(jacobi,
                Pointwise(FloatNear(ferr), expect_jacobi));
}


TEST(NonRigidPCATest, ResidualAndJacobians_2) {
    int numLmks = 2;
    dim3 lmkThrds(BLOCKSIZE);
    dim3 lmkBlocks((numLmks + BLOCKSIZE - 1) / BLOCKSIZE);

    int npoints = 3;
    const float position[] = {1,1,1, 2,2,2, 3,3,3};
    const int lmks[] = {1,2};
    const float deformB[] = {1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7, 8,8,8, 9,9,9};
    int deformB_row = 3;
    int deformB_col = 3*npoints;
    int nScanPnts = 3;
    const float scanPoints[] = {1,1,1, 4,4,4, 6,6,6};
    const int scanLmks[] = {0,2};
    const bool isJacobianRequired = true;

    const float expect_res = 15.0;
    const float expect_jacobi[] = {21.0, 39.0, 57.0};

    float res;
    float jacobi[3];
    float *res_d, *jacobi_d;
    cudaMalloc((void**)&res_d, sizeof(float));
    cudaMalloc((void**)&jacobi_d, deformB_col * sizeof(float));

    float *position_d, *deformB_d, *scanPoints_d;

    CUDA_CHECK(cudaMalloc((void**)&position_d, 3 * npoints * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&deformB_d, deformB_row * deformB_col * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&scanPoints_d, 3 * nScanPnts * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(position_d, position, 3 * npoints * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deformB_d, deformB, deformB_row * deformB_col * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scanPoints_d, scanPoints, 3 * nScanPnts * sizeof(float), cudaMemcpyHostToDevice));

    int *lmks_d, *scanLmks_d;
    CUDA_CHECK(cudaMalloc((void**)&lmks_d, numLmks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&scanLmks_d, numLmks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(lmks_d, lmks, numLmks * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scanLmks_d, scanLmks, numLmks * sizeof(float), cudaMemcpyHostToDevice));

    _calculateLandmarkLoss << < lmkBlocks, lmkThrds >> >(res_d, jacobi_d, position_d,
            deformB_d, deformB_row, deformB_col, lmks_d, scanPoints_d, scanLmks_d, numLmks, isJacobianRequired);
    CHECK_ERROR_MSG("Kernel Error");

    CUDA_CHECK(cudaMemcpy(&res, res_d, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(jacobi, jacobi_d, deformB_col*sizeof(float), cudaMemcpyDeviceToHost));

//    printf("Resudual: %.6f\n", res);
//    printf("Jacobians: [0]=%.6f [1]=%.6f [2]=%.6f\n", jacobi[0], jacobi[1], jacobi[2]);

    float ferr = 1e-5;
    EXPECT_NEAR(res, expect_res, ferr);
    EXPECT_THAT(jacobi,
                Pointwise(FloatNear(ferr), expect_jacobi));
}