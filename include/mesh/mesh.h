#pragma once
#include <vector>
#include <Eigen/Core>

#include "type.h"

namespace telef::mesh {
    using ColorMesh = struct ColorMesh {
        Eigen::VectorXf position;
        Eigen::VectorXf uv;
        telef::types::ImagePtrT image;
        std::vector<uint8_t> color;
        std::vector<std::vector<int>> triangles;

        void applyTransform(Eigen::MatrixXf transform);
    };
}