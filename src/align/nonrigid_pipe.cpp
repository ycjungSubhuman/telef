#include <boost/make_shared.hpp>
#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <Eigen/Core>
#include <cmath>

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
            :isModelInitialized(false) {
        if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas could not be initialized");
        }
    }

    PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(const PCAGPUNonRigidFittingPipe &that) {
        this->isModelInitialized = false;
        if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas could not be initialized");
        }
    }

    PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe(PCAGPUNonRigidFittingPipe &&that) noexcept {
        this->isModelInitialized = that.isModelInitialized;
        this->c_deformModel = that.c_deformModel;
        this->cublasHandle = that.cublasHandle;
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
    PCAGPUNonRigidFittingPipe::_processData(boost::shared_ptr<PCARigidAlignmentSuite> in) {
        /* Load data to cuda device */

        std::cout << "Fitting PCA model GPU" << std::endl;
        if(!isModelInitialized) {
            auto deformBasis = in->pca_model->getBasisMatrix();
            auto ref = in->pca_model->getReferenceVector();
            auto mean = in->pca_model->getMeanDeformation();
            auto landmarks = in->pca_model->getLandmarks();
            loadModelToCUDADevice(&this->c_deformModel, deformBasis, ref, mean, landmarks);
            isModelInitialized = true;
        }

        std::cout << "Initializing Frame Data for GPU fitting" << std::endl;
        // Filter out non-detected Deformable Model landmarks
        std::vector<int> validLmks = in->pca_model->getLandmarks();
        auto riter = in->fittingSuite->invalid3dLandmarks.rbegin();
        pcl::io::savePLYFile("captured.ply", *in->rawCloud);
        while (riter != in->fittingSuite->invalid3dLandmarks.rend())
        {
            auto iter_data = validLmks.begin() + *riter;
            validLmks.erase(iter_data);
            riter++;
        }

        in->transformation = Eigen::Matrix4f::Identity();

        C_ScanPointCloud c_scanPointCloud;
        loadScanToCUDADevice(&c_scanPointCloud, in->rawCloud, in->fittingSuite->rawCloudLmkIdx,
                             validLmks, in->transformation);

        /* Setup Optimizer */

        std::cout << "Fitting PCA model to scan..." << std::endl;
        auto cost = new PCAGPULandmarkDistanceFunctor<SHAPE_RANK>(this->c_deformModel, c_scanPointCloud, cublasHandle);
        ceres::Problem problem;
        double coeff[SHAPE_RANK] = {0,};
        double t[3] = {0.0,0.0, 0.1};
        double u[3] = {0.0,};
        problem.AddResidualBlock(cost, new ceres::CauchyLoss(0.5), coeff, t, u);
        problem.AddResidualBlock(new RegularizerFunctor(SHAPE_RANK, 0.0001), NULL, coeff);
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 1000;

        /* Run Optimization */
        auto summary = ceres::Solver::Summary();
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;

        float fu[3];
        float ft[3];
        float r[9];
        float trans[16];
        convertArray(t, ft, 3);
        convertArray(u, fu, 3);
        calc_r_from_u(r, fu);
        create_trans_from_tu(trans, ft, r);
        Eigen::Map<Eigen::Matrix4f> eigenTrans(trans);

        /* Free Resources */
        freeScanCUDA(c_scanPointCloud);

        auto result = boost::make_shared<PCANonRigidFittingResult>();

        result->fitCoeff = Eigen::Map<Eigen::VectorXd>(coeff, SHAPE_RANK).cast<float>() * 1;

        std::cout << "Fitted: " << std::endl << result->fitCoeff << std::endl;
        result->image = in->image;
        result->pca_model = in->pca_model;
        result->fx = in->fx;
        result->fy = in->fy;
        result->transformation = eigenTrans * in->transformation;

        return result;
    }

}
