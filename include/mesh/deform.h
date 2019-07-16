#pragma once

#include <Eigen/Dense>
#include "type.h"

namespace telef::mesh
{
    Eigen::MatrixXf lmk2deformed(
        const Eigen::MatrixXf &V, 
        const Eigen::MatrixXi &F,
        telef::types::CloudConstPtrT landmark3d,
        const std::vector<int> &lmkinds,
        const float lmkweight=1.0f);
}