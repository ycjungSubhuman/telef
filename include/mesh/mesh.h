#pragma once
#include <vector>
#include <Eigen/Core>

namespace telef::mesh {
    using ColorMesh = struct ColorMesh {
        Eigen::VectorXf position;
        Eigen::VectorXf color;
        std::vector<std::vector<int>> triangles;
    };
}