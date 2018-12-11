#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "solver/costFunction.h"
#include "solver/gpu/cuda/cu_solver.h"

#include "face/model.h"
#include "face/model_cudahelper.h"
#include "face/cu_model_kernel.h"
#include "type.h"
#include "util/convert_arr.h"
#include "util/cudautil.h"

#define TRANSLATE_COEFF 3
#define ROTATE_COEFF 3

namespace {
    using namespace telef::face;
    using namespace telef::types;
    using namespace telef::util;
}


namespace telef::align{
    class PCALandmarkCudaFunction : public telef::solver::CostFunction {
    public:
        //TODO: If you are going to copy or move this object, implement 'the rule of 5'
        PCALandmarkCudaFunction(C_PcaDeformModel c_deformModel, C_ScanPointCloud c_scanPointCloud,
                cublasHandle_t cublasHandle) :
                              telef::solver::CostFunction(c_scanPointCloud.numLmks*3, {}),
                              cublasHandle(cublasHandle),
                              c_deformModel(c_deformModel),
                              c_scanPointCloud(c_scanPointCloud),
                              weight(1.f)
        {
            allocPositionCUDA(&position_d, c_deformModel.dim);
//            parameterSizes.push_back(c_deformModel.shapeRank);
//            parameterSizes.push_back(c_deformModel.expressionRank);
            parameterSizes.push_back(TRANSLATE_COEFF);
            parameterSizes.push_back(ROTATE_COEFF);

            // TODO: Move to helper
            point_pair = {
                    .mesh_position_d=nullptr,
                    .mesh_positoin_before_transform_d=nullptr,
                    .ref_position_d=c_scanPointCloud.scanLandmark_d,
                    .mesh_corr_inds_d=nullptr,
                    .ref_corr_inds_d=nullptr,
                    .point_count=c_scanPointCloud.numLmks
            };

            CUDA_MALLOC(&point_pair.mesh_position_d, static_cast<size_t>(c_deformModel.dim));
            CUDA_MALLOC(&point_pair.mesh_positoin_before_transform_d, static_cast<size_t>(c_deformModel.dim));
            CUDA_MALLOC(&point_pair.mesh_corr_inds_d, static_cast<size_t>(c_scanPointCloud.numLmks));
            CUDA_MALLOC(&point_pair.ref_corr_inds_d, static_cast<size_t>(c_scanPointCloud.numLmks));
        }

        virtual ~PCALandmarkCudaFunction() {
            freePositionCUDA(position_d);

            CUDA_FREE(point_pair.mesh_position_d);
            CUDA_FREE(point_pair.mesh_positoin_before_transform_d);
            CUDA_FREE(point_pair.mesh_corr_inds_d);
            CUDA_FREE(point_pair.ref_corr_inds_d);
        }

        virtual void evaluate(solver::ResidualBlock::Ptr residualBlock) {
//            auto fa1Params = residualBlock->getParameterBlocks()[0];
//            auto fa2Params = residualBlock->getParameterBlocks()[1];
//            auto ftParams = residualBlock->getParameterBlocks()[2];
//            auto fuParams = residualBlock->getParameterBlocks()[3];
            auto ftParams = residualBlock->getParameterBlocks()[0];
            auto fuParams = residualBlock->getParameterBlocks()[1];

            float* fa1Params;
            CUDA_ALLOC_AND_ZERO(&fa1Params, static_cast<size_t >(c_deformModel.shapeRank));
            float* fa2Params;
            CUDA_ALLOC_AND_ZERO(&fa2Params, static_cast<size_t >(c_deformModel.expressionRank));

            C_Residuals c_residuals;
            c_residuals.residual_d = residualBlock->getResiduals();
            c_residuals.numResuduals = residualBlock->numResiduals();

            C_Params c_params;
            c_params.fa1Params_d = fa1Params;
            c_params.numa1 = c_deformModel.shapeRank;

            c_params.fa2Params_d = fa2Params;
            c_params.numa2 = c_deformModel.expressionRank;

            c_params.ftParams_d = ftParams->getParameters();
            c_params.numt = ftParams->numParameters();
            c_params.ftParams_h = new float[c_params.numt];
            CUDA_CHECK(cudaMemcpy(c_params.ftParams_h, ftParams->getParameters(), c_params.numt * sizeof(float), cudaMemcpyDeviceToHost));
            printf("ftParams_h[%d]: %.5f\n", 0, c_params.ftParams_h[0]);
            c_params.fuParams_d = fuParams->getParameters();
            c_params.numu = fuParams->numParameters();
            c_params.fuParams_h = new float[c_params.numu];
            CUDA_CHECK(cudaMemcpy(c_params.fuParams_h, fuParams->getParameters(), c_params.numu * sizeof(float), cudaMemcpyDeviceToHost));
            printf("fuParams_h[%d]: %.5f\n", 0, c_params.fuParams_h[0]);
            CUDA_CHECK(cudaMemset(c_residuals.residual_d, 0, c_residuals.numResuduals*sizeof(float)));

            calculatePointPairs(point_pair, position_d, cublasHandle,
                                c_params, c_deformModel, c_scanPointCloud);

            calculatePointPairLossCuda(c_residuals, point_pair, weight);

            delete[] c_params.ftParams_h;
            delete[] c_params.fuParams_h;

            CUDA_FREE(fa1Params);
            CUDA_FREE(fa2Params);
        }

        virtual void computeJacobians(solver::ResidualBlock::Ptr residualBlock) {
//            auto fa1Params = residualBlock->getParameterBlocks()[0];
//            auto fa2Params = residualBlock->getParameterBlocks()[1];
//            auto ftParams = residualBlock->getParameterBlocks()[2];
//            auto fuParams = residualBlock->getParameterBlocks()[3];
            auto ftParams = residualBlock->getParameterBlocks()[0];
            auto fuParams = residualBlock->getParameterBlocks()[1];

            float* fa1Params;
            CUDA_ALLOC_AND_ZERO(&fa1Params, static_cast<size_t >(c_deformModel.shapeRank));
            float* fa2Params;
            CUDA_ALLOC_AND_ZERO(&fa2Params, static_cast<size_t >(c_deformModel.expressionRank));
//
//            C_Residuals c_residuals;
//            c_residuals.residual_d = residualBlock->getResiduals();
//            c_residuals.numResuduals = residualBlock->numResiduals();

            C_Params c_params;
            c_params.fa1Params_d = fa1Params;
            c_params.numa1 = c_deformModel.shapeRank;

            c_params.fa2Params_d = fa2Params;
            c_params.numa2 = c_deformModel.expressionRank;;

            c_params.ftParams_d = ftParams->getParameters();
            c_params.numt = ftParams->numParameters();
            //TODO: use GPU all the way through, !c_params.ftParams_h
            c_params.ftParams_h = new float[c_params.numt];
            CUDA_CHECK(cudaMemcpy(c_params.ftParams_h, ftParams->getParameters(), c_params.numt * sizeof(float), cudaMemcpyDeviceToHost));

            c_params.fuParams_d = fuParams->getParameters();
            c_params.numu = fuParams->numParameters();
            //TODO: use GPU all the way through, !c_params.fuParams_h
            c_params.fuParams_h = new float[c_params.numu];
            CUDA_CHECK(cudaMemcpy(c_params.fuParams_h, fuParams->getParameters(), c_params.numu * sizeof(float), cudaMemcpyDeviceToHost));

            C_Jacobians c_jacobians;
//            c_jacobians.fa1Jacobian_d = fa1Params->getJacobians();
//            c_jacobians.numa1j = residualBlock->numResiduals() * fa1Params->numParameters();
//
//            c_jacobians.fa2Jacobian_d = fa2Params->getJacobians();
//            c_jacobians.numa2j = residualBlock->numResiduals() * fa2Params->numParameters();

            c_jacobians.ftJacobian_d = ftParams->getJacobians();
            c_jacobians.numtj = residualBlock->numResiduals() * ftParams->numParameters();

            c_jacobians.fuJacobian_d = fuParams->getJacobians();
            c_jacobians.numuj = residualBlock->numResiduals() * fuParams->numParameters();

            CUDA_CHECK(cudaMemset(c_jacobians.ftJacobian_d, 0, c_jacobians.numtj*sizeof(float)));
            CUDA_CHECK(cudaMemset(c_jacobians.fuJacobian_d, 0, c_jacobians.numuj*sizeof(float)));

            //zeroJacobiansCUDA(c_jacobians);
            calculatePointPairDerivatives(c_jacobians, point_pair, c_params, c_deformModel, weight );

            delete[] c_params.ftParams_h;
            delete[] c_params.fuParams_h;


            CUDA_FREE(fa1Params);
            CUDA_FREE(fa2Params);
        }

    protected:
        cublasHandle_t cublasHandle;
        C_PcaDeformModel c_deformModel;
        C_ScanPointCloud c_scanPointCloud;
        float *position_d;

        const float weight;

        PointPair point_pair;
    };

    class L2RegularizerFunctorCUDA : public telef::solver::CostFunction {
    public:
        L2RegularizerFunctorCUDA(int coeffSize, double multiplier)
        : telef::solver::CostFunction(coeffSize, {coeffSize}),
        coeffSize(coeffSize),
        multiplier(multiplier)
        {}

        virtual ~L2RegularizerFunctorCUDA(){}

        virtual void evaluate(solver::ResidualBlock::Ptr residualBlock) {
            float* params_d = residualBlock->getParameterBlocks()[0]->getParameters();
//            float* jacobi_d = residualBlock->getParameterBlocks()[0]->getJacobians();

            // TODO: Reimplement in cuda
            float* parameters = new float[coeffSize];
            float* residuals = new float[coeffSize]{0.0,};

            cudaMemcpy(parameters, params_d, coeffSize * sizeof(float), cudaMemcpyDeviceToHost);
            double sqrt_lambda = sqrt(multiplier);
            for (int i=0; i<coeffSize; i++) {
                residuals[i] = sqrt_lambda*parameters[i];
            }

            // TODO: Reimplement in cuda
            cudaMemcpy(residualBlock->getResiduals(), residuals, coeffSize * sizeof(float), cudaMemcpyHostToDevice);

            delete[] parameters;
            delete[] residuals;

            parameters = NULL;
            residuals = NULL;
        }


        virtual void computeJacobians(solver::ResidualBlock::Ptr residualBlock) {
//            float* params_d = residualBlock->getParameterBlocks()[0]->getParameters();
            float* jacobi_d = residualBlock->getParameterBlocks()[0]->getJacobians();

            // TODO: Reimplement in cuda
            float* jacobians = new float[coeffSize*coeffSize]{0.0,};

            double sqrt_lambda = sqrt(multiplier);

            for(int i=0; i<coeffSize; i++) {
                for (int j=0; j<coeffSize; j++) {
                    jacobians[coeffSize*i + j] = 0;
                }
                jacobians[coeffSize*i + i] = sqrt_lambda;
            }

            cudaMemcpy(jacobi_d, jacobians, coeffSize * coeffSize* sizeof(float), cudaMemcpyHostToDevice);

            delete[] jacobians;
            jacobians = NULL;
        }
    private:
        int coeffSize;
        double multiplier;
    };

    class LinearBarrierFunctorCUDA : public telef::solver::CostFunction {
    public:
        LinearBarrierFunctorCUDA(int coeffSize, double multiplier, double barrierSlope)
                : telef::solver::CostFunction(coeffSize, {coeffSize}),
                  coeffSize(coeffSize),
                  multiplier(multiplier),
                  barrierSlope(barrierSlope) {}

        virtual ~LinearBarrierFunctorCUDA(){}

        virtual void evaluate(solver::ResidualBlock::Ptr residualBlock) {
            float* params_d = residualBlock->getParameterBlocks()[0]->getParameters();

            // TODO: Reimplement in cuda
            float* parameters = new float[coeffSize];
            float* residuals = new float[coeffSize]{0.0,};

            cudaMemcpy(parameters, params_d, coeffSize * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < coeffSize; i++) {
                const double x = parameters[i];
                if(x < 0) {
                    residuals[i] = sqrt(multiplier * (-barrierSlope * x));

                }
                else {
                    residuals[i] = 0;
                }
            }

            cudaMemcpy(residualBlock->getResiduals(), residuals, coeffSize * sizeof(float), cudaMemcpyHostToDevice);


            delete[] parameters;
            delete[] residuals;

            parameters = NULL;
            residuals = NULL;
        }

        virtual void computeJacobians(solver::ResidualBlock::Ptr residualBlock) {
            float* residuals_d = residualBlock->getResiduals();
            float* jacobi_d = residualBlock->getParameterBlocks()[0]->getJacobians();

            // TODO: Reimplement in cuda
            float* residuals = new float[coeffSize]{0.0,};
            float* jacobians = new float[coeffSize*coeffSize]{0.0,};

            cudaMemcpy(residuals, residuals_d, coeffSize * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < coeffSize; i++) {
                if(residuals[i] < 0) {
                    for(int j=0; j<coeffSize; j++) {
                        jacobians[coeffSize*i + j] = 0.0;
                    }
                    jacobians[coeffSize*i + i] = 0.5 * (1.0/residuals[i]) * (-multiplier*barrierSlope);
                }
                else {
                    for(int j=0; j<coeffSize; j++) {
                        jacobians[coeffSize*i + j] = 0.0;
                    }
                }
            }

            cudaMemcpy(jacobi_d, jacobians, coeffSize * coeffSize* sizeof(float), cudaMemcpyHostToDevice);

            delete[] residuals;
            delete[] jacobians;

            residuals = NULL;
            jacobians = NULL;
        }

    private:
        int coeffSize;
        double multiplier;
        double barrierSlope;
    };

    class LinearUpperBarrierFunctorCUDA : public telef::solver::CostFunction {
    public:
        LinearUpperBarrierFunctorCUDA(int coeffSize, double multiplier, double barrierSlope, double barrier)
                : telef::solver::CostFunction(coeffSize, {coeffSize}),
                  coeffSize(coeffSize),
                  multiplier(multiplier),
                  barrierSlope(barrierSlope),
                  barrier(barrier) {}

        virtual ~LinearUpperBarrierFunctorCUDA(){}

        virtual void evaluate(solver::ResidualBlock::Ptr residualBlock) {
            float* params_d = residualBlock->getParameterBlocks()[0]->getParameters();
//            float* jacobi_d = residualBlock->getParameterBlocks()[0]->getJacobians();

            // TODO: Reimplement in cuda
            float* parameters = new float[coeffSize];
            float* residuals = new float[coeffSize]{0.0,};

            cudaMemcpy(parameters, params_d, coeffSize * sizeof(float), cudaMemcpyDeviceToHost);

            //c_residuals.residual_d = residualBlock->getResiduals();
            //c_residuals.numResuduals = residualBlock->numResiduals();
            for (int i = 0; i < coeffSize; i++) {
                const double x = parameters[i];
                if(x > 0) {
                    residuals[i] = sqrt(multiplier * (barrierSlope * x));
                }
                else {
                    residuals[i] = 0;

                }
            }

            cudaMemcpy(residualBlock->getResiduals(), residuals, coeffSize * sizeof(float), cudaMemcpyHostToDevice);

            delete[] parameters;
            delete[] residuals;

            parameters = NULL;
            residuals = NULL;
        }

        virtual void computeJacobians(solver::ResidualBlock::Ptr residualBlock) {
//            float* params_d = residualBlock->getParameterBlocks()[0]->getParameters();
            float* residuals_d = residualBlock->getResiduals();
            float* jacobi_d = residualBlock->getParameterBlocks()[0]->getJacobians();

            // TODO: Reimplement in cuda
            float* residuals = new float[coeffSize]{0.0,};
            float* jacobians = new float[coeffSize*coeffSize]{0.0,};

            cudaMemcpy(residuals, residuals_d, coeffSize * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < coeffSize; i++) {
                if(residuals[i] > 0) {
                    for(int j=0; j<coeffSize; j++) {
                        jacobians[coeffSize*i + j] = 0.0;
                    }
                    jacobians[coeffSize*i + i] = 0.5 * (1.0/residuals[i]) * (multiplier*barrierSlope);
                }
                else {
                    for(int j=0; j<coeffSize; j++) {
                        jacobians[coeffSize*i + j] = 0.0;
                    }
                }
            }

            cudaMemcpy(jacobi_d, jacobians, coeffSize * coeffSize* sizeof(float), cudaMemcpyHostToDevice);

            delete[] residuals;
            delete[] jacobians;

            residuals = NULL;
            jacobians = NULL;
        }

    private:
        int coeffSize;
        double multiplier;
        double barrierSlope;
        double barrier;
    };

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
