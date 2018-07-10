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
    template<unsigned long CoeffRank>
    class PCAGPULandmarkDistanceFunctor : public ceres::CostFunction {
    public:
        //TODO: If you are going to copy are move this object, implement 'rule of 5'
        PCAGPULandmarkDistanceFunctor(C_PcaDeformModel c_deformModel, C_ScanPointCloud c_scanPointCloud,
                                      cublasHandle_t cublasHandle) :
                c_deformModel(c_deformModel),
                c_scanPointCloud(c_scanPointCloud),
                cublasHandle(cublasHandle)
        {
            allocParamsToCUDADevice(&c_params, CoeffRank, TRANSLATE_COEFF, ROTATE_COEFF);
            allocPositionCUDA(&position_d, c_deformModel.dim);
            set_num_residuals(c_scanPointCloud.numLmks*3);
            mutable_parameter_block_sizes()->push_back(CoeffRank);
            mutable_parameter_block_sizes()->push_back(TRANSLATE_COEFF);
            mutable_parameter_block_sizes()->push_back(ROTATE_COEFF);
            fresiduals = new float[c_scanPointCloud.numLmks*3];
            faParams = new float[CoeffRank];
            ftParams = new float[TRANSLATE_COEFF];
            fuParams = new float[ROTATE_COEFF];
            faJacobians = new float[c_scanPointCloud.numLmks*3*CoeffRank];
            ftJacobians = new float[c_scanPointCloud.numLmks*3*TRANSLATE_COEFF];
            fuJacobians = new float[c_scanPointCloud.numLmks*3*ROTATE_COEFF];
        }

        virtual ~PCAGPULandmarkDistanceFunctor() {
            freeParamsCUDA(c_params);
            freePositionCUDA(position_d);

            delete[] fresiduals;
            delete[] faParams;
            delete[] ftParams;
            delete[] fuParams;
            delete[] faJacobians;
            delete[] ftJacobians;
            delete[] fuJacobians;
        }

        virtual bool Evaluate(double const* const* parameters,
                              double* residuals,
                              double** jacobians) const {

            // According to ceres-solver documentation, jacobians and jacobians[i] can be null depending on
            // optimization methods. if either is null, we don't have to compute jacobian
            // FIXME: Change to indicate which one to calcualte
            bool isJacobianRequired = jacobians != nullptr && (jacobians[0] != nullptr || jacobians[1] != nullptr || jacobians[2] != nullptr);
            std::cout << "isJacobianRequired? " << isJacobianRequired << std::endl;
            // Copy to float array
            convertArray(parameters[0], faParams, CoeffRank);
            convertArray(parameters[1], ftParams, TRANSLATE_COEFF);
            convertArray(parameters[2], fuParams, ROTATE_COEFF);

            updateParams(c_params, faParams, CoeffRank, ftParams, TRANSLATE_COEFF, fuParams, ROTATE_COEFF);

            calculateLoss(fresiduals, faJacobians, ftJacobians, fuJacobians, position_d, cublasHandle,
                          c_params, c_deformModel, c_scanPointCloud, isJacobianRequired);

            // Copy back to double array
            convertArray(fresiduals, residuals, c_scanPointCloud.numLmks*3);

            if (isJacobianRequired) {
                convertArray(faJacobians, jacobians[0], c_scanPointCloud.numLmks*3*CoeffRank);
                convertArray(ftJacobians, jacobians[1], c_scanPointCloud.numLmks*3*TRANSLATE_COEFF);
                convertArray(fuJacobians, jacobians[2], c_scanPointCloud.numLmks*3*ROTATE_COEFF);
            }

            return true;
        }
    private:
        float *fresiduals;
        float *faParams;
        float *ftParams;
        float *fuParams;
        float *faJacobians;
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
