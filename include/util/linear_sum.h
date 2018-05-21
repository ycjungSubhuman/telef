#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

namespace telef::util {

    /** Vector generation from linear sum of basis */
    class LinearSumVectorGenerator {
    protected:
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basisMatrix;

    public:
        LinearSumVectorGenerator(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basisMatrix);
        int getRank() const;
        Eigen::MatrixXf getBasisMatrix() const;

        virtual Eigen::VectorXf genVector(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff) const = 0;
        virtual Eigen::VectorXf genVector(const float * const coeff, int size) const = 0;
    };

    class CPULinearSumVectorGenerator : public LinearSumVectorGenerator {
    public:
        CPULinearSumVectorGenerator(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basisMatrix);

        Eigen::VectorXf genVector(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff) const override;
        Eigen::VectorXf genVector(const float * const coeff, int size) const override;
    };

    class GPULinearSumVectorGenerator : public LinearSumVectorGenerator {
    public:
        GPULinearSumVectorGenerator(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basisMatrix);

        Eigen::VectorXf genVector(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff) const override;
        Eigen::VectorXf genVector(const float * const coeff, int size) const override;
    };
}
