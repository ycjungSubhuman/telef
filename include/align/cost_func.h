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
    class PCAGPULandmarkDistanceFunctor : public ceres::CostFunction {
    public:
        //TODO: If you are going to copy or move this object, implement 'the rule of 5'
        PCAGPULandmarkDistanceFunctor(C_PcaDeformModel c_deformModel, C_ScanPointCloud c_scanPointCloud,
                                      cublasHandle_t cublasHandle) :
                c_deformModel(c_deformModel),
                c_scanPointCloud(c_scanPointCloud),
                cublasHandle(cublasHandle)
        {
            allocParamsToCUDADevice(&c_params,
                                    c_deformModel.shapeRank, c_deformModel.expressionRank,
                                    TRANSLATE_COEFF, ROTATE_COEFF);
            allocPositionCUDA(&position_d, c_deformModel.dim);
            set_num_residuals(c_scanPointCloud.numLmks*3);
            mutable_parameter_block_sizes()->push_back(c_deformModel.shapeRank);
            mutable_parameter_block_sizes()->push_back(c_deformModel.expressionRank);
            mutable_parameter_block_sizes()->push_back(TRANSLATE_COEFF);
            mutable_parameter_block_sizes()->push_back(ROTATE_COEFF);
            fresiduals = new float[c_scanPointCloud.numLmks*3];
            fa1Params = new float[c_deformModel.shapeRank];
            fa2Params = new float[c_deformModel.expressionRank];
            ftParams = new float[TRANSLATE_COEFF];
            fuParams = new float[ROTATE_COEFF];
            fa1Jacobians = new float[c_scanPointCloud.numLmks*3*c_deformModel.shapeRank];
            fa2Jacobians = new float[c_scanPointCloud.numLmks*3*c_deformModel.expressionRank];
            ftJacobians = new float[c_scanPointCloud.numLmks*3*TRANSLATE_COEFF];
            fuJacobians = new float[c_scanPointCloud.numLmks*3*ROTATE_COEFF];
        }

        virtual ~PCAGPULandmarkDistanceFunctor() {
            freeParamsCUDA(c_params);
            freePositionCUDA(position_d);

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
            std::cout << "isJacobianRequired? " << isJacobianRequired << std::endl;
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

            calculateLoss(fresiduals, fa1Jacobians, fa2Jacobians, ftJacobians, fuJacobians, position_d, cublasHandle,
                          c_params, c_deformModel, c_scanPointCloud, isJacobianRequired);

            // Copy back to double array
            convertArray(fresiduals, residuals, c_scanPointCloud.numLmks*3);

            if (isJacobianRequired) {
                convertArray(fa1Jacobians, jacobians[0], c_scanPointCloud.numLmks*3*c_deformModel.shapeRank);
                convertArray(fa2Jacobians, jacobians[1], c_scanPointCloud.numLmks*3*c_deformModel.expressionRank);
                convertArray(ftJacobians, jacobians[2], c_scanPointCloud.numLmks*3*TRANSLATE_COEFF);
                convertArray(fuJacobians, jacobians[3], c_scanPointCloud.numLmks*3*ROTATE_COEFF);
            }

            return true;
        }
    private:
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
    };

    class RegularizerFunctor : public ceres::CostFunction {
    public:
        RegularizerFunctor(int coeffSize, double multiplier) : coeffSize(coeffSize), multiplier(multiplier) {
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
                        if(i==j) {
                            jacobians[0][coeffSize*j+i] = sqrt_lambda;
                        }
                        else {
                            jacobians[0][coeffSize*j+i] = 0;
                        }
                    }
                }
            }
            return true;
        }
    private:
        int coeffSize;
        double multiplier;
    };
}
