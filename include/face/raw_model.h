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
    int numPoints;
} C_ScanPointCloud;

typedef struct C_Params {
    float *params_d;
    int numParams;
} C_Params;


