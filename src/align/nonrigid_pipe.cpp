#include <boost/make_shared.hpp>
#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <Eigen/Core>
#include <cmath>

#include "solver/gpu/gpuSolver.h"
#include "solver/gpu/gpuProblem.h"
#include "solver/costFunction.h"
#include "solver/util/cudautil.h"

#include "align/nonrigid_pipe.h"
#include "align/cost_func.h"
#include "face/model_cudahelper.h"
#include "type.h"
#include "util/transform.h"
#include "util/cu_quaternion.h"
#include "util/convert_arr.h"

#define EPS 0.005

using namespace telef::types;
using namespace telef::face;

namespace telef::align {
    PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe()
            :isModelInitialized(false),
             geoWeight(0), geoMaxPoints(0), geoSearchRadius(0), addGeoTerm(false)
    {
        if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas could not be initialized");
        }
    }

    PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(const float geoWeight, const int geoMaxPoints,
                                                         const float geoSearchRadius, const bool addGeoTerm)
            :isModelInitialized(false),
             geoWeight(geoWeight), geoMaxPoints(geoMaxPoints), geoSearchRadius(geoSearchRadius), addGeoTerm(addGeoTerm)
    {
        if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas could not be initialized");
        }
    }

    PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(const PCAGPUNonRigidFittingPipe &that) {
        this->isModelInitialized = false;
        this->geoWeight = that.geoWeight;
        this->geoMaxPoints = that.geoMaxPoints;
        this->geoSearchRadius = that.geoSearchRadius;
        this->addGeoTerm = that.addGeoTerm;
        if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas could not be initialized");
        }
    }

    PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(PCAGPUNonRigidFittingPipe &&that) noexcept {
        this->isModelInitialized = that.isModelInitialized;
        this->c_deformModel = that.c_deformModel;
        this->cublasHandle = that.cublasHandle;
        this->geoWeight = that.geoWeight;
        this->geoMaxPoints = that.geoMaxPoints;
        this->geoSearchRadius = that.geoSearchRadius;
        this->addGeoTerm = that.addGeoTerm;
    }

    PCAGPUNonRigidFittingPipe& PCAGPUNonRigidFittingPipe::operator=(const PCAGPUNonRigidFittingPipe &that) {
        if(&that != this) {
            if(isModelInitialized) {
                freeModelCUDA(c_deformModel);
                this->isModelInitialized = false;
            }
            cublasDestroy(cublasHandle);
            if (cublasCreate(&this->cublasHandle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Cublas could not be initialized");
            }
        }

        this->geoWeight = that.geoWeight;
        this->geoMaxPoints = that.geoMaxPoints;
        this->geoSearchRadius = that.geoSearchRadius;
        this->addGeoTerm = that.addGeoTerm;
        return *this;
    }

    PCAGPUNonRigidFittingPipe& PCAGPUNonRigidFittingPipe::operator=(PCAGPUNonRigidFittingPipe &&that) {
        if(&that != this) {
            if(isModelInitialized) {
                freeModelCUDA(c_deformModel);
                this->isModelInitialized = false;
            }
            cublasDestroy(cublasHandle);
            this->isModelInitialized = that.isModelInitialized;
            this->c_deformModel = that.c_deformModel;
            this->cublasHandle = that.cublasHandle;
            this->geoWeight = that.geoWeight;
            this->geoMaxPoints = that.geoMaxPoints;
            this->geoSearchRadius = that.geoSearchRadius;
            this->addGeoTerm = that.addGeoTerm;
        }

        return *this;
    }

    PCAGPUNonRigidFittingPipe::~PCAGPUNonRigidFittingPipe() {
        if(isModelInitialized) {
            freeModelCUDA(c_deformModel);
        }
        cublasDestroy(cublasHandle);
    }

    boost::shared_ptr<PCANonRigidFittingResult>
    PCAGPUNonRigidFittingPipe::_processData(boost::shared_ptr<PCANonRigidAlignmentSuite> in) {
        /* Load data to cuda device */

        //std::cout << "Fitting PCA model GPU" << std::endl;
        if(!isModelInitialized) {
            auto shapeBasis = in->pca_model->getShapeBasisMatrix();
            auto expressionBasis = in->pca_model->getExpressionBasisMatrix();
            auto ref = in->pca_model->getReferenceVector();
            auto meanShapeDeformation = in->pca_model->getShapeDeformationCenter();
            auto meanExpressionDeformation = in->pca_model->getExpressionDeformationCenter();
            auto landmarks = in->pca_model->getLandmarks();
            loadModelToCUDADevice(&this->c_deformModel, shapeBasis, expressionBasis, ref,
                                  meanShapeDeformation, meanExpressionDeformation, landmarks);
            isModelInitialized = true;
        }

        C_ScanPointCloud c_scanPointCloud;
        loadScanToCUDADevice(&c_scanPointCloud, in->rawCloud, in->fx, in->fy, landmarkSelection,
                             in->transformation, in->fittingSuite->landmark3d);

        /* Setup Optimizer */
        //std::cout << "Fitting PCA model to scan..." << std::endl;

        auto solver = std::make_shared<solver::GPUSolver>();
        auto problem = std::make_shared<solver::GPUProblem>();

        std::vector<int> nParams = {2};
        int nRes = 4;
        auto lmkcost = std::make_shared<PCALandmarkCudaFunction>(this->c_deformModel, c_scanPointCloud, cublasHandle);
        auto l2lmkReg = std::make_shared<L2RegularizerFunctorCUDA>(c_deformModel.shapeRank, 0.002);
        auto lBarrierExpReg = std::make_shared<LinearBarrierFunctorCUDA>(c_deformModel.expressionRank, 0.02, 10);
        auto lUBarrierExpReg = std::make_shared<LinearUpperBarrierFunctorCUDA>(c_deformModel.expressionRank, 0.002, 2, 1.0);

        float *shapeCoeff = new float[c_deformModel.shapeRank]{0,};
        float *expressionCoeff = new float[c_deformModel.expressionRank]{0,};
        float ft[3] = {0.0,};
        float fu[3] = {3.14, 0.0, 0.0};
        std::vector<float*> initParams = {shapeCoeff, expressionCoeff, ft, fu};

        solver::ResidualFunction::Ptr lmkFunc = problem->addResidualFunction(lmkcost, initParams);
        solver::ResidualFunction::Ptr l2LmkRegFunc = problem->addResidualFunction(l2lmkReg, {shapeCoeff});
        solver::ResidualFunction::Ptr lBarrierExpRegFunc = problem->addResidualFunction(lBarrierExpReg, {expressionCoeff});
        solver::ResidualFunction::Ptr lUBarrierExpRegFunc = problem->addResidualFunction(lUBarrierExpReg, {expressionCoeff});
        l2LmkRegFunc->getResidualBlock()->getParameterBlocks()[0]->share(lmkFunc->getResidualBlock()->getParameterBlocks()[0]);
        lBarrierExpRegFunc->getResidualBlock()->getParameterBlocks()[0]->share(lmkFunc->getResidualBlock()->getParameterBlocks()[1]);
        lUBarrierExpRegFunc->getResidualBlock()->getParameterBlocks()[0]->share(lmkFunc->getResidualBlock()->getParameterBlocks()[1]);

        solver->options.max_iterations = 200;
        solver->options.verbose = true;
        solver->options.target_error_change = 1e-8;
        solver->options.lambda_initial = 1e-1;
        solver->options.step_down = 10;
        solver->options.step_up = 2;

        solver->solve(problem);

//        auto lmkCost = new PCAGPULandmarkDistanceFunctor(this->c_deformModel, c_scanPointCloud, cublasHandle);

//        ceres::Problem problem;
//        double *shapeCoeff = new double[c_deformModel.shapeRank]{0,};
//        double *expressionCoeff = new double[c_deformModel.expressionRank]{0,};
//        double t[3] = {0.0,};
//        double u[3] = {3.14, 0.0, 0.0};
//        problem.AddResidualBlock(lmkCost, new ceres::CauchyLoss(0.5), shapeCoeff, expressionCoeff, t, u);

//        if (addGeoTerm == true) {
//            auto geoCost = new PCAGPUGeometricDistanceFunctor(this->c_deformModel, c_scanPointCloud, cublasHandle,
//                    geoMaxPoints*3,sqrtf(geoWeight), geoSearchRadius);
//            problem.AddResidualBlock(geoCost, new ceres::CauchyLoss(0.5), shapeCoeff, expressionCoeff, t, u);
//        }
//        problem.AddResidualBlock(new L2RegularizerFunctor(c_deformModel.shapeRank, 0.0002), NULL, shapeCoeff);
//        problem.AddResidualBlock(new LinearBarrierFunctor(c_deformModel.expressionRank, 0.0002, 10), NULL, expressionCoeff);
//        problem.AddResidualBlock(new LinearUpperBarrierFunctor(c_deformModel.expressionRank, 0.00002, 2, 1.0), NULL, expressionCoeff);
//        ceres::Solver::Options options;
//        options.minimizer_progress_to_stdout = false;
//        options.max_num_iterations = 1000;
//        options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;
//        options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
//        options.line_search_direction_type = ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT;
//        options.line_search = ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT;
//        options.nonlinear_conjugate_gradient_type = ceres::NonlinearConjugateGradientType::FLETCHER_REEVES;
//        options.line_search_type = ceres::LineSearchType::WOLFE;

        /* Run Optimization */
//        auto summary = ceres::Solver::Summary();
//        ceres::Solve(options, &problem, &summary);
//        std::cout << summary.FullReport() << std::endl;

//        float fu[3];
//        float ft[3];
        float r[9];
        float trans[16];
//        convertArray(t, ft, 3);
//        convertArray(u, fu, 3);
        calc_r_from_u(r, fu);
        create_trans_from_tu(trans, ft, r);
        Eigen::Map<Eigen::Matrix4f> eigenTrans(trans);

        /* Free Resources */
        freeScanCUDA(c_scanPointCloud);

        auto result = boost::make_shared<PCANonRigidFittingResult>();
        result->shapeCoeff =
                Eigen::Map<Eigen::VectorXf>(shapeCoeff, c_deformModel.shapeRank)/*.cast<float>()*/;
        result->expressionCoeff =
                Eigen::Map<Eigen::VectorXf>(expressionCoeff, c_deformModel.expressionRank)/*.cast<float>()*/;

        std::cout << "Fitted(Shape): " << std::endl;
        std::cout << result->shapeCoeff << std::endl;
        std::cout << "Fitted(Expression): " << std::endl;
        std::cout << result->expressionCoeff << std::endl;
        for (int i = 0; i < 3; i++){
            printf("u[%d]:%.5f\n", i, fu[i]);
        }

        for (int i = 0; i < 3; i++){
            printf("t[%d]:%.5f\n", i, ft[i]);
        }

        result->image = in->image;
        result->pca_model = in->pca_model;
        result->cloud = in->rawCloud;
        result->landmark3d = in->fittingSuite->landmark3d;
        result->fx = in->fx;
        result->fy = in->fy;
        result->transformation = eigenTrans * in->transformation;

        delete[] shapeCoeff;
        delete[] expressionCoeff;

        return result;
    }

    const std::vector<int> PCAGPUNonRigidFittingPipe::landmarkSelection {
        5, 6, 7, 8, 9, 10, 11,      // Chin
        // Others(frontal part)
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    };
}
