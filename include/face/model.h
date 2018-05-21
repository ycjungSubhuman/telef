#pragma once

#include <string>
#include <vector>
#include <memory>
#include <experimental/filesystem>
#include <exception>
#include <algorithm>
#include <tuple>
#include <functional>
#include <random>
#include <ctime>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include "io/ply/meshio.h"
#include "type.h"
#include "util/eigen_pcl.h"
#include "util/linear_sum.h"

namespace {
    using namespace telef::util;
}

namespace telef::face {

    /** PCA model for deformations */
    class PCADeformationModel {
    private:
        int shapeRank;
        std::shared_ptr<LinearSumVectorGenerator> linearSumGen;

    public:
        Eigen::VectorXf mean;

        PCADeformationModel() = default;
        PCADeformationModel(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> shapeBase, Eigen::VectorXf mean,
                            bool useGPU=false);


        explicit PCADeformationModel(std::vector<ColorMesh> &samples, ColorMesh &refMesh, int shapeRank,
                                     bool useGPU=false);

        Eigen::VectorXf genDeform(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff) const;

        Eigen::VectorXf genDeform(const float * const coeff, int size) const;

        int getRank() const ;

        Eigen::MatrixXf getBasisMatrix() const ;

    };

    /** PCA face model using PCA model of deformation btw reference mesh and samples*/
    class MorphableFaceModel {
    private:
        PCADeformationModel deformModel;
        ColorMesh refMesh;
        std::vector<int> landmarks;
        std::random_device rd;
        std::mt19937 mt;
        int shapeRank;
    public:
        /** Construct PCA Model using a list of mesh files */
        MorphableFaceModel(std::vector<fs::path> &f, int shapeRank, bool rigidAlignRequired=false, bool useGPU=false);

        /** Load from existing model file */
        MorphableFaceModel(fs::path fileName, bool useGPU=false);

        /** Save this model to a file */
        void save(fs::path fileName);

        /* Generate a xyzxyz... position vector using given coefficients */
        Eigen::VectorXf genPosition(Eigen::VectorXf shapeCoeff) const;

        Eigen::VectorXf genPosition(const float * const shapeCoeff, int size) const;

        ColorMesh genMesh(const float * const shapeCoeff, int size) const;

        /* Generate a ColorMesh using given coefficients */
        ColorMesh genMesh(Eigen::VectorXf shapeCoeff) const;

        Eigen::VectorXf getBasis(unsigned long coeffIndex) const;

        int getRank() const;

        /* Generate a random sample ColorMesh */
        ColorMesh sample(bool printDebug=false);

        void setLandmarks(std::vector<int> lmk);

        std::vector<int> getLandmarks() const;
    };
};
