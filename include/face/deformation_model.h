#pragma once

#include <iostream>
#include <experimental/filesystem>

#include <Eigen/Core>
#include <Eigen/SVD>
#include "mesh/mesh.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::mesh;

    template <int Rank>
    Eigen::Matrix<float, Eigen::Dynamic, Rank> getPCABase(Eigen::MatrixXf data) {

        // Each row is a data. This is to match data matrix dimensions with formulas in Wikipedia.
        auto d = data.transpose();
        Eigen::MatrixXf centered = d.rowwise() - d.colwise().mean();
        // Fast singlular value computation using devide-and-conquer
        Eigen::BDCSVD<Eigen::MatrixXf> bdc(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Sort eigenvectors according to (singular value)^2 / (n -1), which is equal to eigenvalues
        std::vector<std::pair<float, Eigen::VectorXf>> pairs;
        if (d.rows() <= d.cols()) { //singular values are shorter than position dimension
            std::cout << "Singular values are shorter then dimension" << std::endl;
            pairs.resize(static_cast<unsigned long>(bdc.singularValues().rows()));
        }
        else { // singular values are exact match with V
            std::cout << "Exact match" << std::endl;
            pairs.resize(static_cast<unsigned long>(d.cols()));
        }
        std::cout << "Singluar Value **2" << std::endl;
        for(unsigned long i=0; i<pairs.size(); i++) {
            auto s = bdc.singularValues()(i);
            auto s2 = s*s;
            std::cout << s2 << ", ";
            pairs[i] = std::make_pair(
                    s2, // propertional to eigenvalue (omitted /(n-1))
                    bdc.matrixV().col(i)); // eivenvector, which is a PCA basis
        }
        std::cout << std::endl;
        std::sort(pairs.begin(), pairs.end(), [](auto &l, auto &r) {return l.first > r.first;});

        Eigen::Matrix<float, Eigen::Dynamic, Rank> result(d.cols(), Rank);
        for (int i = 0; i < std::min(Rank, static_cast<int>(bdc.singularValues().rows())); i++) {
            result.col(i).swap(pairs[i].second);
        }

        // Fill in empty basis if given PCA rank is higher than the rank of the data matrix
        if (Rank > d.cols()) {
            std::cout << "WARNING : given rank is higher than number of singular values" << std::endl;
            for (long i = d.cols(); i < Rank; i++) {
                result.col(i) = Eigen::VectorXf::Zero(d.cols());
            }
        }
        return result;
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
