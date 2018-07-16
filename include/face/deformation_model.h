#pragma once

#include <iostream>
#include <experimental/filesystem>
#include <Eigen/Core>
#include <Eigen/SVD>
#include "mesh/mesh.h"

namespace {
    using namespace telef::mesh;
}

namespace telef::face {
    /** PCA model for deformations */
    class PCADeformationModel {
    public:
        Eigen::MatrixXf pcaBasisVectors;
        Eigen::VectorXf mean;
        int rank;

        PCADeformationModel() = default;
        PCADeformationModel(Eigen::MatrixXf shapeBase, Eigen::VectorXf mean, int rank);

        PCADeformationModel(std::vector<ColorMesh> &samples, ColorMesh &refMesh, int rank);

        Eigen::VectorXf genDeform(Eigen::VectorXf coeff);

        Eigen::VectorXf genDeform(const double * const coeff, int size);
    };
}
