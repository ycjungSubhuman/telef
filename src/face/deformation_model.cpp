#include "face/deformation_model.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "io/matrix.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::mesh;
    using namespace telef::io;

    Eigen::MatrixXf getPCABase(Eigen::MatrixXf data, int maxRank) {
        std::cout << "Calculating PCA Basis..." << std::endl;
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

        Eigen::MatrixXf result(d.cols(), maxRank);
        for (int i = 0; i < std::min(maxRank, static_cast<int>(bdc.singularValues().rows())); i++) {
            result.col(i).swap(pairs[i].second);
        }

        // Fill in empty basis if given PCA rank is higher than the rank of the data matrix
        if (maxRank > d.cols()) {
            std::cout << "WARNING : given rank is higher than number of singular values" << std::endl;
            for (long i = d.cols(); i < maxRank; i++) {
                result.col(i) = Eigen::VectorXf::Zero(d.cols());
            }
        }
        std::cout << "PCA Basis Calculation Done" << std::endl;
        return result;
    }

    Eigen::VectorXf linearSum(const Eigen::MatrixXf basisMatrix, const Eigen::VectorXf center,
                              const double *coeff, int rank) {
        if(rank != basisMatrix.cols()) {
            throw std::runtime_error("Coefficient dimension mismatch");
        }
        Eigen::VectorXf result = Eigen::VectorXf::Zero(basisMatrix.rows());
        for (long i=0; i<rank; i++) {
            result += coeff[i] * basisMatrix.col(i);
        }
        return center + result;
    }
}

namespace telef::face {

    int PCADeformationModel::getRank() {
        return rank;
    }

    Eigen::VectorXf PCADeformationModel::getCenter() {
        return mean;
    }

    Eigen::MatrixXf PCADeformationModel::getBasisMatrix() {
        return pcaBasisVectors;
    }

    PCADeformationModel::PCADeformationModel(const std::vector<ColorMesh> &samples, const ColorMesh &refMesh, int rank) {
        pcaBasisVectors.resize(samples.size(), rank);
        auto numData = samples.size();
        auto dataDim = refMesh.position.size();
        Eigen::MatrixXf positions(dataDim, numData);
        Eigen::MatrixXf colors(dataDim, numData);

        for(unsigned long i=0; i<samples.size(); i++) {
            auto mesh = samples[i];
            positions.col(i) = mesh.position.col(0) - refMesh.position.col(0);
        }

        pcaBasisVectors = getPCABase(positions, rank);
        mean = positions.rowwise().mean();
        this->rank = rank;
    }

    PCADeformationModel::PCADeformationModel(fs::path path) {
        readMat((path.string()+".base").c_str(), pcaBasisVectors);
        readMat((path.string()+".mean").c_str(), mean);
        this->rank = static_cast<int>(pcaBasisVectors.cols());
    }

    Eigen::VectorXf PCADeformationModel::genDeform(Eigen::VectorXf coeff) {
        Eigen::VectorXd c = coeff.cast<double>();
        return linearSum(pcaBasisVectors, mean, c.data(), rank);
    }

    Eigen::VectorXf PCADeformationModel::genDeform(const double *coeff, int size) {
        assert(size==rank);
        return linearSum(pcaBasisVectors, mean, coeff, rank);
    }

    void PCADeformationModel::save(fs::path path) {
        writeMat((path.string()+".base").c_str(), pcaBasisVectors);
        writeMat((path.string()+".mean").c_str(), mean);
    }

    BlendShapeDeformationModel::BlendShapeDeformationModel(const std::vector<ColorMesh> &samples,
                                                           const ColorMesh &refMesh, int rank) {
        assert(samples.size() == rank && samples.size() > 0);
        assert(refMesh.position.size() == samples[0].position.size());
        blendShapeVectors.resize(refMesh.position.size(), rank);
        for(int i=0; i<rank; i++) {
            Eigen::VectorXf deformation = samples[i].position - refMesh.position;
            std::copy_n(deformation.data(),
                        deformation.size(),
                        blendShapeVectors.data()+i*refMesh.position.size());
        }
        this->rank = rank;
    }

    BlendShapeDeformationModel::BlendShapeDeformationModel(fs::path path) {
        readMat((path.string()+".base").c_str(), blendShapeVectors);
        this->rank = static_cast<int>(blendShapeVectors.cols());
    }

    int BlendShapeDeformationModel::getRank() {
        return rank;
    }

    Eigen::VectorXf BlendShapeDeformationModel::getCenter() {
        return Eigen::VectorXf::Zero(blendShapeVectors.rows());
    }

    Eigen::MatrixXf BlendShapeDeformationModel::getBasisMatrix() {
        return blendShapeVectors;
    }

    Eigen::VectorXf BlendShapeDeformationModel::genDeform(Eigen::VectorXf coeff) {
        Eigen::VectorXd c = coeff.cast<double>();
        return linearSum(blendShapeVectors, getCenter(), c.data(), rank);
    }

    Eigen::VectorXf BlendShapeDeformationModel::genDeform(const double *coeff, int size) {
        assert(size==rank);
        return linearSum(blendShapeVectors, getCenter(), coeff, rank);
    }

    void BlendShapeDeformationModel::save(fs::path path) {
        writeMat((path.string()+".base").c_str(), blendShapeVectors);
    }
}
