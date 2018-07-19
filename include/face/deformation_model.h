#pragma once

#include <iostream>
#include <experimental/filesystem>
#include <Eigen/Core>
#include <Eigen/SVD>
#include "mesh/mesh.h"

namespace {
    using namespace telef::mesh;
    namespace fs = std::experimental::filesystem;
}

namespace telef::face {
    class LinearModel {
    public:
        virtual int getRank()=0;
        virtual Eigen::VectorXf getCenter()=0;
        virtual Eigen::MatrixXf getBasisMatrix()=0;
        virtual Eigen::VectorXf genDeform(Eigen::VectorXf coeff)=0;
        virtual Eigen::VectorXf genDeform(const double *coeff, int size)=0;
        virtual void save(fs::path path)=0;
    };

    /** PCA model for deformations */
    class PCADeformationModel : public LinearModel {
    public:
        PCADeformationModel(const std::vector<ColorMesh> &samples, const ColorMesh &refMesh, int rank);

        explicit PCADeformationModel(fs::path path);

        int getRank() override;

        Eigen::VectorXf getCenter() override;

        Eigen::MatrixXf getBasisMatrix() override;

        Eigen::VectorXf genDeform(Eigen::VectorXf coeff) override;

        Eigen::VectorXf genDeform(const double *coeff, int size) override;

        void save(fs::path path) override;
    private:
        Eigen::MatrixXf pcaBasisVectors;
        Eigen::VectorXf mean;
        int rank;
    };

    /** Simple blend shape of deformations reletive to reference mesh */
    class BlendShapeDeformationModel : public LinearModel {
    public:
        BlendShapeDeformationModel(const std::vector<ColorMesh> &samples, const ColorMesh &refMesh, int rank);
        explicit BlendShapeDeformationModel(fs::path path);
        int getRank() override;
        Eigen::VectorXf getCenter() override;
        Eigen::MatrixXf getBasisMatrix() override;
        Eigen::VectorXf genDeform(Eigen::VectorXf coeff) override;
        Eigen::VectorXf genDeform(const double *coeff, int size) override;
        void save(fs::path path) override;
    private:
        Eigen::MatrixXf blendShapeVectors;
        int rank;
    };
}
