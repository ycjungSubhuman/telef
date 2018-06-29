#pragma once

/** Each struct resides in host memory, while pointer members live in device memory */
typedef struct C_PcaDeformModel {
    float *deformBasis_d;
    float *ref_d;
    int *lmks_d;

    int lmkCount;
    int rank;
    int dim;
} C_PcaDeformModel;

typedef struct C_ScanPointCloud {
    float *scanPoints_d;
    float *rigidTransform_d;
    int *validModelLmks_d;
    int *scanLmks_d;

    int numPoints;
    // Transformation Matrix dims
    int transformCols;
    int transformRows;
    // Size of valid lmks and scan lmk points should be same
    int numLmks;
} C_ScanPointCloud;

typedef struct C_Params {
    float *params_d;
    int numParams;
} C_Params;
