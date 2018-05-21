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

namespace telef::face {

    /** PCA model for deformations */
    class PCADeformationModel {
    private:
	int shapeRank;
    public:
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> shapeBase;
        Eigen::VectorXf mean;

        PCADeformationModel() = default;
        PCADeformationModel(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> shapeBase, Eigen::VectorXf mean);


        explicit PCADeformationModel(std::vector<ColorMesh> &samples, ColorMesh &refMesh, int shapeRank);

        Eigen::VectorXf genDeform(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff);

        Eigen::VectorXf genDeform(const double * const coeff, int size);
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
        MorphableFaceModel(std::vector<fs::path> &f, int shapeRank, bool rigidAlignRequired=false);;

        /** Load from existing model file */
        MorphableFaceModel(fs::path fileName);

        /** Save this model to a file */
        void save(fs::path fileName);

        /* Generate a xyzxyz... position vector using given coefficients */
        Eigen::VectorXf genPosition(Eigen::VectorXf shapeCoeff);

        Eigen::VectorXf genPosition(const double * const shapeCoeff, int size);

        ColorMesh genMesh(const double * const shapeCoeff, int size);

        /* Generate a ColorMesh using given coefficients */
        ColorMesh genMesh(Eigen::VectorXf shapeCoeff);

        Eigen::VectorXf getBasis(unsigned long coeffIndex);

        int getRank();

        /* Generate a random sample ColorMesh */
        ColorMesh sample();

        void setLandmarks(std::vector<int> lmk);

        std::vector<int> getLandmarks();
    };
};
