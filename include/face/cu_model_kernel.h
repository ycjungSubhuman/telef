#pragma once

#include <cuda_runtime_api.h>
#include "face/raw_model.h"

__global__
void _calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);
void calculateVertexPosition(float *position_d, const C_Params params, const C_PcaDeformModel deformModel);
void calculateLoss(float *residual, float *jacobian,
                   const float *position_d, const C_Params params,
                   const C_PcaDeformModel deformModel, const C_ScanPointCloud scanPointCloud);
