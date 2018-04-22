#pragma once
#include <vector>
#include <Eigen/Core>

namespace telef::mesh {
    using ColorMesh = struct ColorMesh {
        Eigen::VectorXf position;
        std::vector<uint8_t> color;
        std::vector<std::vector<int>> triangles;

        void applyTransform(Eigen::Matrix4f transform);
    };
}