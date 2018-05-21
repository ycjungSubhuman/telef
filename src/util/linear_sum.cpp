#include "util/linear_sum.h"
#include "util/linear_sum_gpu.h"

namespace telef::util {
    LinearSumVectorGenerator::LinearSumVectorGenerator(
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basisMatrix) :
        basisMatrix(basisMatrix) {}

    int LinearSumVectorGenerator::getRank() const {
        return static_cast<int>(basisMatrix.cols());
    }

	Eigen::MatrixXf LinearSumVectorGenerator::getBasisMatrix() const {
        return basisMatrix;
	}

	CPULinearSumVectorGenerator::CPULinearSumVectorGenerator(
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basisMatrix) : LinearSumVectorGenerator(basisMatrix) {}

    Eigen::VectorXf CPULinearSumVectorGenerator::genVector(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff) const {
        return genVector(coeff.data(), coeff.size());
    }

    Eigen::VectorXf CPULinearSumVectorGenerator::genVector(const float *const coeff, int size) const {
        if(size != basisMatrix.cols()) {
            throw std::runtime_error("Coefficient dimension mismatch");
        }
        Eigen::VectorXf result = Eigen::VectorXf::Zero(basisMatrix.rows());
        for (long i=0; i<size; i++) {
            result += coeff[i] * basisMatrix.col(i);
        }
        return result;
    }

    GPULinearSumVectorGenerator::GPULinearSumVectorGenerator(
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basisMatrix) : LinearSumVectorGenerator(basisMatrix) {}

    Eigen::VectorXf GPULinearSumVectorGenerator::genVector(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff) const {
        return genVector(coeff.data(), coeff.size());
    }

    Eigen::VectorXf GPULinearSumVectorGenerator::genVector(const float *const coeff, int size) const {
        assert(basisMatrix.cols() == size);
        Eigen::VectorXf result(basisMatrix.rows());
        const int M = static_cast<int>(basisMatrix.rows());
        const int N = static_cast<int>(basisMatrix.cols());

        getLinearSum(basisMatrix.data(), M, coeff, N, result.data());

        return result;
    }
}
