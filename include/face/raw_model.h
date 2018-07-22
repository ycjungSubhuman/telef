#pragma once

/** Each struct resides in host memory, while pointer members live in device memory */
typedef struct C_PcaDeformModel {
    float *shapeDeformBasis_d;
    float *expressionDeformBasis_d;
    float *ref_d;
    float *meanShapeDeformation_d;
    float *meanExpressionDeformation_d;
    int *lmks_d;

    int lmkCount;
    int shapeRank;
    int expressionRank;
    int dim;
} C_PcaDeformModel;

typedef struct C_ScanPointCloud {
    float *scanPoints_d;
    float *scanLandmark_d;
    float *rigidTransform_d;
    unsigned int width;
    unsigned int height;
    float fx;
    float fy;
    float cx;
    float cy;

    int *modelLandmarkSelection_d;
    int numPoints;

    // Size of valid lmks and scan lmk points should be same
    int numLmks;
} C_ScanPointCloud;

typedef struct C_Params {
    float *fa1Params_d; //used for shape parameters
    float *fa2Params_d; //used for expression parameters
    float *ftParams_d;
    float *fuParams_d;

    float *ftParams_h;
    float *fuParams_h;

    int numa1;
    int numa2;
    int numt;
    int numu;
} C_Params;
