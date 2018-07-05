#include <gtest/gtest.h>
#include <cstdlib>
#include <cblas.h>
#include <math.h>
#include "align/cu_loss.h"
#include "util/cudautil.h"
#include "../testutil/diff.h"

/** Calculate vertex position given basis and coefficients */
// it is in src/face/cu_model_kernels.cu
void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);

static C_PcaDeformModel get_mock_model(const float *fake_basis, int rank, int dim, const int *fake_lmks, int num_lmk) {
    C_PcaDeformModel model;
    //Setup PCA basis
    CUDA_CHECK(cudaMalloc((void**)(&model.deformBasis_d), rank*dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(model.deformBasis_d, fake_basis, rank*dim*sizeof(float), cudaMemcpyHostToDevice));
    model.rank = rank;
    model.dim = dim;

    //Setup Landmarks
    // Always use the first point as landmark 0
    CUDA_CHECK(cudaMalloc((void**)(&model.lmks_d), num_lmk*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(model.lmks_d, fake_lmks, num_lmk*sizeof(int), cudaMemcpyHostToDevice));
    model.lmkCount = num_lmk;

    //Setup Reference Mesh
    float *fakeRef = (float*)malloc(dim*sizeof(float));
    float *fakeMean = (float*)malloc(dim*sizeof(float));
    for (int i=0; i<dim; i++) {
        fakeRef[i] = 0.0f;
        fakeMean[i] = 0.0f;
    }

    CUDA_CHECK(cudaMalloc((void**)(&model.ref_d), dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(model.ref_d, fakeRef, dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)(&model.mean_d), dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(model.mean_d, fakeMean, dim*sizeof(float), cudaMemcpyHostToDevice));
    free(fakeRef);
    free(fakeMean);

    return model;
}

static void free_model(C_PcaDeformModel model) {
    CUDA_CHECK(cudaFree(model.deformBasis_d));
    CUDA_CHECK(cudaFree(model.lmks_d));
    CUDA_CHECK(cudaFree(model.ref_d));
    CUDA_CHECK(cudaFree(model.mean_d));
}

static C_ScanPointCloud get_mock_scan(const float *fake_scan, int num_points, const int *fake_lmks, int num_lmk) {
    C_ScanPointCloud scan;
    //Setup scan points
    CUDA_CHECK(cudaMalloc((void**)(&scan.scanPoints_d), num_points*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(scan.scanPoints_d, fake_scan, num_points*3*sizeof(float), cudaMemcpyHostToDevice));
    scan.numPoints = num_points;

    //Setup Landmark
    // Always use the first point as landmark 0
    scan.numLmks = 1;
    CUDA_CHECK(cudaMalloc((void**)(&scan.scanLmks_d), num_lmk*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(scan.scanLmks_d, fake_lmks, num_lmk*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)(&scan.validModelLmks_d), num_lmk*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(scan.validModelLmks_d, fake_lmks, num_lmk*sizeof(int), cudaMemcpyHostToDevice));

    return scan;
}

static void free_scan(C_ScanPointCloud scan) {
    CUDA_CHECK(cudaFree(scan.scanPoints_d));
    CUDA_CHECK(cudaFree(scan.scanLmks_d));
    CUDA_CHECK(cudaFree(scan.validModelLmks_d));
}


static void test_lmk_loss(float oracle,
                          const float *position, int num_position_points,
                          const float *scan_positions, int num_scan_points, const int *lmks, int num_lmks) {
    float mse_h;
    float *mse_d;

    CUDA_CHECK(cudaMalloc((void**)(&mse_d), sizeof(float)));

    float *position_d;
    CUDA_CHECK(cudaMalloc((void**)(&position_d), num_position_points*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(position_d, position, num_position_points*3*sizeof(float), cudaMemcpyHostToDevice));
    C_ScanPointCloud scan = get_mock_scan(scan_positions, num_scan_points, lmks, num_lmks);

    calc_mse_lmk(mse_d, position_d, scan);
    CUDA_CHECK(cudaMemcpy(&mse_h, mse_d, sizeof(float), cudaMemcpyDeviceToHost));

    free_scan(scan);
    CUDA_CHECK(cudaFree(position_d));
    ASSERT_FLOAT_EQ(oracle, mse_h);
}

// We use these global variables for test_lmk_derivatives
// At the each call of test_lmk_derivatives, they are initialized
// At the end of each call of test_lmk_derivatives, they are destroyed
// The reason we are using global variables is to keep the signature of '_calc_loss' to void(float*, const float*),
// while providing these data to '_calc_loss'
C_ScanPointCloud _glob_scan;
C_PcaDeformModel _glob_model;

static void calc_position(float *position, C_PcaDeformModel model,
                          const float *t, const float *u, const float *a, bool apply_rigid_transform=true) {
    C_Params a_params;
    a_params.numParams = model.rank;
    CUDA_CHECK(cudaMalloc((void**)(&a_params.params_d), model.rank*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a_params.params_d, a, model.rank*sizeof(float), cudaMemcpyHostToDevice));
    float *position_before_transform_d;
    float *position_before_transform_h = (float*)malloc(model.dim*sizeof(float));
    CUDA_CHECK(cudaMalloc((void**)(&position_before_transform_d), model.dim*sizeof(float)));
    calculateVertexPosition(position_before_transform_d, a_params, model);
    if(!apply_rigid_transform) {
        CUDA_CHECK(cudaMemcpy(position, position_before_transform_d,
                              model.dim*sizeof(float), cudaMemcpyDeviceToHost));
        return;
    }
    CUDA_CHECK(cudaMemcpy(position_before_transform_h, position_before_transform_d,
                          model.dim*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(a_params.params_d));
    CUDA_CHECK(cudaFree(position_before_transform_d));

    // calculate rotation
    float rotation[9];
    calc_r_from_u(rotation, u);

    printf ("u: %f %f %f\n", u[0], u[1], u[2]);

    printf("rotation: \n");
    for(int i=0; i<3; i++) {
        printf("%f %f %f\n", rotation[3*i], rotation[3*i+1], rotation[3*i+2]);
    }

    // apply rotation
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                3, model.dim/3, 3, 1.0f,
                rotation, 3,
                position_before_transform_h, 3,
                0.0f, position, 3);
    free(position_before_transform_h);
    // apply transpose
    printf("position: \n");
    for(int i=0; i<model.dim; i++) {
        position[i] += t[i%3];
        printf("%f\n", position[i]);
    }
}

void _calc_loss(float *value, const float *param) {
    // Load parameters
    const float *t = param;
    const float *u = param+3;
    const float *a = param+6;

    float *position = (float*)malloc(_glob_model.dim*sizeof(float));
    calc_position(position, _glob_model, t, u, a);

    // calculate loss
    // setup position_d for 'calc_mse_lmk'
    float *position_d;
    float *value_d;
    CUDA_CHECK(cudaMalloc((void**)(&value_d), sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&position_d), _glob_model.dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(position_d, position, _glob_model.dim*sizeof(float), cudaMemcpyHostToDevice));
    printf("position_d: %f %f %f\n", position[0], position[1], position[2]);
    free(position);
    calc_mse_lmk(value_d, position_d, _glob_scan);
    // copy back the result to host
    CUDA_CHECK(cudaMemcpy(value, value_d, sizeof(float), cudaMemcpyDeviceToHost));
    // clean up
    CUDA_CHECK(cudaFree(value_d));
    CUDA_CHECK(cudaFree(position_d));
}

static void test_lmk_derivatives(C_PcaDeformModel model, C_ScanPointCloud scan,
                                 const float *t, const float *u, const float *a) {
    const int num_a = model.rank;
    ////////////// Calculate numerical derivative
    float *param = (float*)malloc((3+3+num_a)*sizeof(float));
    float *numerical = (float*)malloc((3+3+num_a)*sizeof(float));
    float *analytic =(float*)malloc((3+3+num_a)*sizeof(float));
    memcpy(param, t, 3*sizeof(float));
    memcpy(param+3, u, 3*sizeof(float));
    memcpy(param+6, a, num_a*sizeof(float));

    // setup global variables for '_calc_loss'
    _glob_model = model;
    _glob_scan = scan;

    // get numerical differentiation
    func f = &_calc_loss;
    printf("NUMERICALLLLLLL\n");
    calc_numerical_diff(numerical, &f, 0.01f, 1, (3+3+num_a), param);

    ////////////// Calculate analytic derivative
    float *de_dt_d;
    float *de_du_d;
    float *de_da_d;
    float *u_d;
    float *position_d;
    float *position_before_transform_d;

    CUDA_CHECK(cudaMalloc((void**)(&de_dt_d), 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&de_du_d), 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&de_da_d), num_a*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&u_d), 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&position_d), model.dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&position_before_transform_d), model.dim*sizeof(float)));

    // calculate mesh vertex position again
    // (we are doing it in _calc_loss, but I didn't want to copy them back from it)
    float *position_before_transform = (float*)malloc(model.dim*sizeof(float));
    float *position = (float*)malloc(model.dim*sizeof(float));
    printf("ANALYTICALLLLLLL\n");
    calc_position(position_before_transform, model, t, u, a, false);
    calc_position(position, model, t, u, a, true);

    CUDA_CHECK(cudaMemcpy(u_d, u, 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(position_d, position, model.dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(position_before_transform_d, position_before_transform, model.dim*sizeof(float), cudaMemcpyHostToDevice));
    calc_derivatives_lmk(de_dt_d, de_du_d, de_da_d, u_d, position_before_transform_d, position_d, model, scan);
    CUDA_CHECK(cudaMemcpy(analytic, de_dt_d, 3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(analytic+3, de_du_d, 3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(analytic+6, de_da_d, num_a*sizeof(float), cudaMemcpyDeviceToHost));

    // clean up
    free(position);
    CUDA_CHECK(cudaFree(de_dt_d));
    CUDA_CHECK(cudaFree(de_du_d));
    CUDA_CHECK(cudaFree(de_da_d));
    CUDA_CHECK(cudaFree(u_d));
    CUDA_CHECK(cudaFree(position_d));
    CUDA_CHECK(cudaFree(position_before_transform_d));
    for(int i=0; i<6+model.rank; i++) {
        printf("%f %f\n", numerical[i], analytic[i]);
    }

    /////////////// Compare btw two
    printf("de_dts\n");
    for(int i=0; i<3; i++) {
        ASSERT_LE(fabs(numerical[i]-analytic[i]), 0.001f);
    }
    printf("de_dus\n");
    for(int i=3; i<6; i++) {
        ASSERT_LE(fabs(numerical[i]-analytic[i]), 0.001f);
    }
    printf("de_das\n");
    for(int i=6; i<6+model.rank; i++) {
        ASSERT_LE(fabs(numerical[i]-analytic[i]), 0.001f);
    }
}

TEST(LmkLoss, ExactlySame) {
    float position[] = {1.0f, 2.0f, 3.0f};
    float scan_positions[] = {1.0f, 2.0f, 3.0f};
    int lmks[] = {0};

    test_lmk_loss(0.0f,
                position, 1,
                scan_positions, 1,
                lmks, 1);
}

TEST(LmkLoss, DifferentByTwo) {
    float position[] = {3.0f, 2.0f, 3.0f};
    float scan_positions[] = {1.0f, 2.0f, 3.0f};
    int lmks[] = {0};

    test_lmk_loss(4.0f,
                position, 1,
                scan_positions, 1,
                lmks, 1);
}

TEST(LmkLoss, ExatlySameTwoPoints) {
    float position[] = {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f};
    float scan_positions[] = {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f};
    int lmks[] = {0, 1};

    test_lmk_loss(0.0f,
                position, 2,
                scan_positions, 2,
                lmks, 2);
}

TEST(LmkLoss, DifferentTwoPoints) {
    float position[] = {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f};
    float scan_positions[] = {1.0f, 2.0f, 1.5f,
                              4.0f, 5.0f, 4.5f};
    int lmks[] = {0, 1};

    test_lmk_loss(2.25f,
                  position, 2,
                  scan_positions, 2,
                  lmks, 2);
}

TEST(LmkLossDerivative, ExactlySame) {
    float basis[] = {1.0f, 2.0f, 1.0f};
    float scan_points[] = {1.0f, 2.0f, 1.0f};
    int lmks[] = {0};
    float t[3] = {0.0f, 0.0f, 0.0f};
    float u[3] = {0.0f, 0.0f, 0.0f};
    float a[3] = {1.0f, 1.0f, 1.0f};

    C_PcaDeformModel model = get_mock_model(basis, 1, 3, lmks, 1);
    C_ScanPointCloud scan = get_mock_scan(scan_points, 1, lmks, 1);
    test_lmk_derivatives(model, scan, t, u, a);
    free_model(model);
    free_scan(scan);
}

TEST(LmkLossDerivative, DifferentByOne) {
    float basis[] = {1.0f, 2.0f, 1.0f};
    float scan_points[] = {1.0f, 3.0f, 2.0f};
    int lmks[] = {0};
    float t[3] = {0.0f, 0.0f, 0.0f};
    float u[3] = {0.0f, 0.5f, 0.5f};
    float a[1] = {1.0f};

    C_PcaDeformModel model = get_mock_model(basis, 1, 3, lmks, 1);
    C_ScanPointCloud scan = get_mock_scan(scan_points, 1, lmks, 1);
    test_lmk_derivatives(model, scan, t, u, a);
    free_model(model);
    free_scan(scan);
}
