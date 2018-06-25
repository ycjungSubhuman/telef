#pragma once

#include <cuda_runtime_api.h>

__device__
void calc_dll_du(float *dll_du_d, const float *u_d);

__device__
void calc_dll_dt(float *dll_du_d, const float *t_d);

