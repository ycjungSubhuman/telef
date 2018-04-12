#pragma once
#include <Eigen>
#include <Eigen/Core>

namespace telef::mesh {
    using ColorMesh = struct ColorMesh {
        Eigen::VectorXf position;
        Eigen::VectorXf color;
        Eigen::Matrix3Xf triangles;
    };
}