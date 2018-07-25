#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "face/model.h"
#include "face/model_cudahelper.h"
#include "face/cu_model_kernel.h"
#include "type.h"
#include "util/convert_arr.h"

#define TRANSLATE_COEFF 3
#define ROTATE_COEFF 3

namespace {
    using namespace telef::face;
    using namespace telef::types;
    using namespace telef::util;
}


namespace telef::align{
    class PCAGPUDistanceFunctor : public ceres::CostFunction {
    public:
        //TODO: If you are going to copy or move this object, implement 'the rule of 5'
        PCAGPUDistanceFunctor(C_PcaDeformModel c_deformModel, C_ScanPointCloud c_scanPointCloud,
                              cublasHandle_t cublasHandle, const float weight, const int numResiduals) :
                c_deformModel(c_deformModel),
                c_scanPointCloud(c_scanPointCloud),
                cublasHandle(cublasHandle),
                weight(weight)
        {
            allocParamsToCUDADevice(&c_params,
                                    c_deformModel.shapeRank, c_deformModel.expressionRank,
                                    TRANSLATE_COEFF, ROTATE_COEFF);
            allocPositionCUDA(&position_d, c_deformModel.dim);
            allocResidualsToCUDADevice(&c_residuals, numResiduals);
            allocJacobiansToCUDADevice(&c_jacobians, numResiduals, c_deformModel.shapeRank, c_deformModel.expressionRank,
                                       TRANSLATE_COEFF, ROTATE_COEFF);
            //TODO: allocate PointPair
            set_num_residuals(numResiduals);
            mutable_parameter_block_sizes()->push_back(c_deformModel.shapeRank);
            mutable_parameter_block_sizes()->push_back(c_deformModel.expressionRank);
            mutable_parameter_block_sizes()->push_back(TRANSLATE_COEFF);
            mutable_parameter_block_sizes()->push_back(ROTATE_COEFF);
            fresiduals = new float[numResiduals];
            fa1Params = new float[c_deformModel.shapeRank];
            fa2Params = new float[c_deformModel.expressionRank];
            ftParams = new float[TRANSLATE_COEFF];
            fuParams = new float[ROTATE_COEFF];
            fa1Jacobians = new float[numResiduals*c_deformModel.shapeRank];
            fa2Jacobians = new float[numResiduals*c_deformModel.expressionRank];
            ftJacobians = new float[numResiduals*TRANSLATE_COEFF];
            fuJacobians = new float[numResiduals*ROTATE_COEFF];
        }

        virtual ~PCAGPUDistanceFunctor() {
            freeParamsCUDA(c_params);
            freePositionCUDA(position_d);
            freeResidualsCUDA(c_residuals);
            freeJacobiansCUDA(c_jacobians);
            //TODO: Free PointPair

            delete[] fresiduals;
            delete[] fa1Params;
            delete[] fa2Params;
            delete[] ftParams;
            delete[] fuParams;
            delete[] fa1Jacobians;
            delete[] fa2Jacobians;
            delete[] ftJacobians;
            delete[] fuJacobians;
        }

        virtual bool Evaluate(double const* const* parameters,
                              double* residuals,
                              double** jacobians) const {

            // According to ceres-solver documentation, jacobians and jacobians[i] can be null depending on
            // optimization methods. if either is null, we don't have to compute jacobian
            // FIXME: Change to indicate which one to calcualte
            bool isJacobianRequired = jacobians != nullptr;
            // Copy to float array
            convertArray(parameters[0], fa1Params, c_deformModel.shapeRank);
            convertArray(parameters[1], fa2Params, c_deformModel.expressionRank);
            convertArray(parameters[2], ftParams, TRANSLATE_COEFF);
            convertArray(parameters[3], fuParams, ROTATE_COEFF);

            updateParams(c_params,
                         fa1Params, c_deformModel.shapeRank,
                         fa2Params, c_deformModel.expressionRank,
                         ftParams, TRANSLATE_COEFF,
                         fuParams, ROTATE_COEFF);

            zeroResidualsCUDA(c_residuals);
            zeroJacobiansCUDA(c_jacobians);

            evaluateLoss(isJacobianRequired);

            // Copy back to double array
            convertArray(fresiduals, residuals, num_residuals());

            if (isJacobianRequired) {
                convertArray(fa1Jacobians, jacobians[0], num_residuals()*c_deformModel.shapeRank);
                convertArray(fa2Jacobians, jacobians[1], num_residuals()*c_deformModel.expressionRank);
                convertArray(ftJacobians, jacobians[2], num_residuals()*TRANSLATE_COEFF);
                convertArray(fuJacobians, jacobians[3], num_residuals()*ROTATE_COEFF);
            }

            return true;
        }

        virtual bool evaluateLoss(const bool isJacobianRequired) const = 0;

    protected:
        float *fresiduals;
        float *fa1Params;
        float *fa2Params;
        float *ftParams;
        float *fuParams;
        float *fa1Jacobians;
        float *fa2Jacobians;
        float *ftJacobians;
        float *fuJacobians;

        cublasHandle_t cublasHandle;
        C_PcaDeformModel c_deformModel;
        C_ScanPointCloud c_scanPointCloud;
        C_Params c_params;
        float *position_d;
        C_Residuals c_residuals;
        C_Jacobians c_jacobians;
        //TODO: PointPair pointPair;

        const float weight;
    };


    class PCAGPULandmarkDistanceFunctor : public PCAGPUDistanceFunctor {
    public:
        PCAGPULandmarkDistanceFunctor(C_PcaDeformModel c_deformModel, C_ScanPointCloud c_scanPointCloud,
                                      cublasHandle_t cublasHandle) :
                PCAGPUDistanceFunctor(c_deformModel, c_scanPointCloud, cublasHandle, 1.f, c_scanPointCloud.numLmks*3)
        {}

        virtual ~PCAGPULandmarkDistanceFunctor() {}

        virtual bool evaluateLoss(const bool isJacobianRequired) const {
            calculateLandmarkLoss(fresiduals, fa1Jacobians, fa2Jacobians, ftJacobians, fuJacobians, position_d,
                                  cublasHandle, c_params, c_deformModel,
                                  c_scanPointCloud, c_residuals, c_jacobians, weight, isJacobianRequired);

            return true;
        }


    };

    class PCAGPUGeometricDistanceFunctor : public PCAGPUDistanceFunctor {
    public:
        PCAGPUGeometricDistanceFunctor(C_PcaDeformModel c_deformModel, C_ScanPointCloud c_scanPointCloud,
                                       cublasHandle_t cublasHandle, const int num_residuals,
                                       const float weight, const float searchRadius) :
                PCAGPUDistanceFunctor(c_deformModel, c_scanPointCloud, cublasHandle, weight, num_residuals),
                searchRadius(searchRadius)
        {}

        virtual ~PCAGPUGeometricDistanceFunctor() {}

        virtual bool evaluateLoss(const bool isJacobianRequired) const {
            calculateGeometricLoss(fresiduals, fa1Jacobians, fa2Jacobians, ftJacobians, fuJacobians, position_d,
                                   cublasHandle, c_params, c_deformModel,
                                   c_scanPointCloud, c_residuals, c_jacobians, searchRadius, weight, isJacobianRequired);

            return true;
        }

    private:
        const float searchRadius;
    };

    class L2RegularizerFunctor : public ceres::CostFunction {
    public:
        L2RegularizerFunctor(int coeffSize, double multiplier) : coeffSize(coeffSize), multiplier(multiplier) {
            set_num_residuals(coeffSize);
            mutable_parameter_block_sizes()->push_back(coeffSize);
        }

        virtual bool Evaluate(double const* const* parameters,
                              double* residuals,
                              double** jacobians) const {
            double sqrt_lambda = sqrt(multiplier);
            for (int i=0; i<coeffSize; i++) {
                residuals[i] = sqrt_lambda*parameters[0][i];
            }
            if(jacobians != nullptr) {
                for(int i=0; i<coeffSize; i++) {
                    for (int j=0; j<coeffSize; j++) {
                        jacobians[0][coeffSize*i + j] = 0;
                    }
                    jacobians[0][coeffSize*i + i] = sqrt_lambda;
                }
            }
            return true;
        }
    private:
        int coeffSize;
        double multiplier;
    };

    /*
     * high linear slope on x < 0
     *                .  |           .
     *                .  |          .
     *                 . |         .
     *                 . |        .
     *                  .|     .
     *                  .| .
     * ---------------------------------------
     *                   |
     *                   |
     *                   |
     *                   |
     *                   |
     *                   |
     *
     */
    class BarrieredL2Functor : public ceres::CostFunction {
    public:
        BarrieredL2Functor(int coeffSize, double multiplier, double barrierSlope)
                : coeffSize(coeffSize),
                  multiplier(multiplier),
                  barrierSlope(barrierSlope) {
            set_num_residuals(coeffSize);
            mutable_parameter_block_sizes()->push_back(coeffSize);
        }

        virtual bool Evaluate(double const* const* parameters,
                              double* residuals,
                              double** jacobians) const {

            for (int i = 0; i < coeffSize; i++) {
                const double x = parameters[0][i];
                if(x < 0) {
                    residuals[i] = sqrt(multiplier * (-barrierSlope * x));

                    if(jacobians != nullptr) {
                        for(int j=0; j<coeffSize; j++) {
                            jacobians[0][coeffSize*i + j] = 0.0;
                        }
                        jacobians[0][coeffSize*i + i] = 0.5 * (1.0/residuals[i]) * (-multiplier*barrierSlope);
                    }
                }
                else {
                    residuals[i] = sqrt(multiplier) * x;

                    if(jacobians != nullptr) {
                        for(int j=0; j<coeffSize; j++) {
                            jacobians[0][coeffSize*i + j] = 0.0;
                        }
                        jacobians[0][coeffSize*i + i] = sqrt(multiplier);
                    }
                }
            }

            return true;
        }

    private:
        int coeffSize;
        double multiplier;
        double barrierSlope;
    };

    /*
     * high linear slope on x < 0
     *                .  |
     *                .  |
     *                 . |
     *                 . |
     *                  .|
     *                  .| . . .  . . . . .
     * ---------------------------------------
     *                   |
     *                   |
     *                   |
     *                   |
     *                   |
     *                   |
     *
     */
    class LinearBarrierFunctor : public ceres::CostFunction {
    public:
        LinearBarrierFunctor(int coeffSize, double multiplier, double barrierSlope)
                : coeffSize(coeffSize),
                  multiplier(multiplier),
                  barrierSlope(barrierSlope) {
            set_num_residuals(coeffSize);
            mutable_parameter_block_sizes()->push_back(coeffSize);
        }

        virtual bool Evaluate(double const* const* parameters,
                              double* residuals,
                              double** jacobians) const {

            for (int i = 0; i < coeffSize; i++) {
                const double x = parameters[0][i];
                if(x < 0) {
                    residuals[i] = sqrt(multiplier * (-barrierSlope * x));

                    if(jacobians != nullptr) {
                        for(int j=0; j<coeffSize; j++) {
                            jacobians[0][coeffSize*i + j] = 0.0;
                        }
                        jacobians[0][coeffSize*i + i] = 0.5 * (1.0/residuals[i]) * (-multiplier*barrierSlope);
                    }
                }
                else {
                    residuals[i] = 0;

                    if(jacobians != nullptr) {
                        for(int j=0; j<coeffSize; j++) {
                            jacobians[0][coeffSize*i + j] = 0.0;
                        }
                    }
                }
            }

            return true;
        }

    private:
        int coeffSize;
        double multiplier;
        double barrierSlope;
    };

    /*
     * high linear slope on x > barrier
     *                   |   .
     *                   |   .
     *                   |  .
     *                   |  .
     *                   | .
     *   ................| .
     * ---------------------------------------
     *                   |
     *                   |
     *                   |
     *                   |
     *                   |
     *                   |
     *
     */
    class LinearUpperBarrierFunctor : public ceres::CostFunction {
    public:
        LinearUpperBarrierFunctor(int coeffSize, double multiplier, double barrierSlope, double barrier)
                : coeffSize(coeffSize),
                  multiplier(multiplier),
                  barrierSlope(barrierSlope),
                  barrier(barrier) {
            set_num_residuals(coeffSize);
            mutable_parameter_block_sizes()->push_back(coeffSize);
        }

        virtual bool Evaluate(double const* const* parameters,
                              double* residuals,
                              double** jacobians) const {

            for (int i = 0; i < coeffSize; i++) {
                const double x = parameters[0][i];
                if(x > 0) {
                    residuals[i] = sqrt(multiplier * (barrierSlope * x));

                    if(jacobians != nullptr) {
                        for(int j=0; j<coeffSize; j++) {
                            jacobians[0][coeffSize*i + j] = 0.0;
                        }
                        jacobians[0][coeffSize*i + i] = 0.5 * (1.0/residuals[i]) * (multiplier*barrierSlope);
                    }
                }
                else {
                    residuals[i] = 0;

                    if(jacobians != nullptr) {
                        for(int j=0; j<coeffSize; j++) {
                            jacobians[0][coeffSize*i + j] = 0.0;
                        }
                    }
                }
            }

            return true;
        }

    private:
        int coeffSize;
        double multiplier;
        double barrierSlope;
        double barrier;
    };
}
