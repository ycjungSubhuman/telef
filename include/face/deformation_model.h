#pragma once

#include <iostream>
#include <experimental/filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include "mesh/mesh.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::mesh;

    template <int Rank>
    Eigen::Matrix<float, Eigen::Dynamic, Rank> getPCABase(Eigen::MatrixXf data) {
        cv::Mat input(data.rows(), data.cols(), CV_32F, data.data());

        std::cout << "Converting shits" << std::endl;
        cv::PCA pca(input, cv::Mat(), 1, Rank);
        std::cout << pca.eigenvectors.rows << "/" << pca.eigenvectors.cols << std::endl;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;
        cv::cv2eigen(pca.eigenvectors, eigenvectors);
        Eigen::Ref<Eigen::Matrix<float, Rank, Eigen::Dynamic>> result(eigenvectors);
        return result.transpose();
    }
}

namespace telef::face {
    /** PCA model for deformations */
    template <int ShapeRank>
    class PCADeformationModel {
    public:
        Eigen::MatrixXf shapeBase;
        Eigen::VectorXf mean;

        PCADeformationModel() = default;
        PCADeformationModel(Eigen::MatrixXf shapeBase, Eigen::VectorXf mean)
        : shapeBase(shapeBase), mean(mean) {}


        explicit PCADeformationModel(std::vector<ColorMesh> &samples, ColorMesh &refMesh)
        {
            shapeBase.resize(samples.size(), ShapeRank);
            auto numData = samples.size();
            auto dataDim = refMesh.position.size();
            Eigen::MatrixXf positions(dataDim, numData);
            Eigen::MatrixXf colors(dataDim, numData);

            for(unsigned long i=0; i<samples.size(); i++) {
                auto mesh = samples[i];
                positions.col(i) = mesh.position.col(0) - refMesh.position.col(0);
            }


            shapeBase = getPCABase<ShapeRank>(positions);
            mean = positions.rowwise().mean();
        }

        Eigen::VectorXf genDeform(Eigen::Matrix<float, ShapeRank, 1> coeff) {
            if(coeff.rows() != shapeBase.cols()) {
                throw std::runtime_error("Coefficient dimension mismatch");
            }
            Eigen::VectorXf result = Eigen::VectorXf::Zero(shapeBase.rows());
            for (long i=0; i<ShapeRank; i++) {
                result += coeff(i) * shapeBase.col(i);
            }
            return mean + result;
        }

        Eigen::VectorXf genDeform(const double * const coeff, int size) {
            if(size != shapeBase.cols()) {
                throw std::runtime_error("Coefficient dimension mismatch");
            }
            Eigen::VectorXf result = Eigen::VectorXf::Zero(shapeBase.rows());
            for (long i=0; i<ShapeRank; i++) {
                result += coeff[i] * shapeBase.col(i);
            }
            return mean + result;
        }

        /**
         * Templated for Ceres AutoDiff, T is eiter a double or Jet
         * @tparam T
         * @param coeff
         * @param size
         * @return
         */
        template <typename T>
        Eigen::Matrix<T, Eigen::Dynamic, 1> genDeformCeres(const T* const coeff, int size) {
            auto shapeBaseT = shapeBase.template cast<T>();
            if(size != shapeBaseT.cols()) {
                throw std::runtime_error("Coefficient dimension mismatch");
            }
            Eigen::Matrix<T, Eigen::Dynamic, 1> result = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(shapeBaseT.rows());
            for (long i=0; i<ShapeRank; i++) {
                result = result + (coeff[i] * shapeBaseT.col(i));
            }

            //cast to T
            return mean.cast<T>() + result;
        }
    };
}
