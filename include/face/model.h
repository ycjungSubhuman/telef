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

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include "io/ply/meshio.h"

namespace fs = std::experimental::filesystem;

using namespace telef::mesh;

namespace {
    template <int Rank>
    Eigen::Matrix<float, Eigen::Dynamic, Rank> getPCABase(Eigen::MatrixXf data) {

        // Each row is a data. This is to match data matrix dimensions with formulas in Wikipedia.
        auto d = data.transpose();
        Eigen::MatrixXf centered = d.colwise() - d.rowwise().mean();
        // Fast singlular value computation using devide-and-conquer
        Eigen::BDCSVD<Eigen::MatrixXf> bdc(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Sort eigenvectors according to (singular value)^2 / (n -1), which is equal to eigenvalues
        std::vector<std::pair<float, Eigen::VectorXf>> pairs;
        if (d.rows() <= d.cols()) { //singular values are shorter than position dimension
            pairs.resize(static_cast<unsigned long>(bdc.singularValues().rows()));
        }
        else { // singular values are exact match with V
            pairs.resize(static_cast<unsigned long>(d.cols()));
        }
        for(unsigned long i=0; i<pairs.size(); i++) {
            pairs[i] = std::make_pair(
                    std::pow(bdc.singularValues()(i), 2.0), // propertional to eigenvalue (omitted /(n-1))
                    bdc.matrixV().col(i)); // eivenvector, which is a PCA basis
        }
        std::sort(pairs.begin(), pairs.end(), [](auto &l, auto &r) {return l.first > r.first;});

        Eigen::Matrix<float, Eigen::Dynamic, Rank> result(d.cols(), Rank);
        std::cout << result.col(0).rows() << " " << result.col(0).cols() << std::endl;
        for (int i = 0; i < std::min(Rank, static_cast<int>(bdc.singularValues().rows())); i++) {
            result.col(i).swap(pairs[i].second);
        }

        // Fill in empty basis if given PCA rank is higher than the rank of the data matrix
        if (Rank > d.cols()) {
            for (long i = d.cols(); i < Rank; i++) {
                result.col(i) = Eigen::VectorXf::Zero(d.cols());
            }
        }
        return result;
    }

    ColorMesh read(fs::path f) {
        if (f.extension().string() == ".ply") {
            return ColorMesh(telef::io::ply::readMesh(f));
        }
        else {
            throw std::runtime_error("File " + f.string() + " is not supported");
        }
    }

    /** PCA model for deformations */
    template <int ShapeRank>
    class PCADeformationModel {
    public:
        Eigen::Matrix<float, Eigen::Dynamic, ShapeRank> shapeBase;
        Eigen::VectorXf mean;

        PCADeformationModel() = default;
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
            coeff = coeff / coeff.sum();
            Eigen::VectorXf result = Eigen::VectorXf::Zero(shapeBase.rows());
            for (long i=0; i<ShapeRank; i++) {
                result += coeff(i) * shapeBase.col(i);
            }
            return mean + result;
        }
    };
}

namespace telef::face {

    /** PCA face model using PCA model of deformation btw reference mesh and samples*/
    template <int ShapeRank>
    class MorphableFaceModel {
    private:
        PCADeformationModel<ShapeRank> deformModel;
        ColorMesh refMesh;
        std::vector<int> landmarks;
        std::random_device rd;
        std::mt19937 mt;
    public:
        MorphableFaceModel(std::vector<fs::path> &f)
        :mt(rd())
        {
            assert(f.size() > 0);
            std::vector<ColorMesh> meshes(f.size());
            std::cout <<"Loading Files" <<std::endl;
            std::transform(f.begin(), f.end(), meshes.begin(), [](auto &a){return read(a);});
            std::cout <<"Loading Finished" << std::endl;

            refMesh = meshes[0];
            deformModel = PCADeformationModel<ShapeRank>(meshes, refMesh);
        }

        /* Generate a ColorMesh using given coefficients */
        ColorMesh genMesh(Eigen::VectorXf shapeCoeff) {
            ColorMesh result;
            result.position = refMesh.position + deformModel.genDeform(shapeCoeff);
            result.triangles = refMesh.triangles;

            return result;
        }

        /* Generate a random sample ColorMesh */
        ColorMesh sample() {
            std::normal_distribution<float> dist;

            std::vector<float> coeff(static_cast<unsigned long>(ShapeRank));
            std::generate(coeff.begin(), coeff.end(), [this, &dist]{return dist(this->mt);});
            return genMesh(Eigen::Map<Eigen::Matrix<float, ShapeRank, 1>>(coeff.data(), coeff.size())*0.05);
        }
    };
}