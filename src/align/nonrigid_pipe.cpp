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

#define EPS 0.005

using namespace telef::types;
using namespace telef::face;

namespace telef::align {
    boost::shared_ptr<PCANonRigidFittingResult>
    PCANonRigidFittingPipe::_processData(boost::shared_ptr<telef::align::PCARigidAlignmentSuite> in)
    {
        auto result = boost::make_shared<PCANonRigidFittingResult>();
        pcl::io::savePLYFile("captured.ply", *in->rawCloud);
        result->pca_model = in->pca_model;

        ceres::LossFunction* loss_function = NULL;
        //ceres::LocalParameterization* quaternion_local_parameterization =
        //        new ceres::EigenQuaternionParameterization;

        ceres::Problem problem;
        auto cost = PCALandmarkDistanceFunctor<RANK>::create(in->pca_model->getLandmarks(), in->fittingSuite->landmark3d,
                                          in->pca_model, in->fittingSuite->invalid3dLandmarks, in->transformation,
                                          1.0);
        double coeff[RANK] = {0,};
        double rigidCoeff[7] = {0,0,0,0,0,0,1.0}; // yaw, pitch, roll, tx, ty, tz, scale
        // The ownership of 'cost' is moved to 'probelm'. So we don't delete cost outsideof 'problem'.
        problem.AddResidualBlock(cost, loss_function, coeff, rigidCoeff);
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 1000;
        options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
        options.function_tolerance = 1e-4;
        //options.use_nonmonotonic_steps = true;
        auto summary = ceres::Solver::Summary();
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        std::cout << coeff[0] << std::endl;
        std::cout << "wat??" << std::endl;

        result->fitCoeff = Eigen::Map<Eigen::VectorXd>(coeff, RANK).cast<float>();
        std::cout << result->fitCoeff << std::endl;
        result->image = in->image;
        result->fx = in->fx;
        result->fy = in->fy;
        result->transformation = in->transformation;

        return result;
    }

    PCAGPUNonRigidFittingPipe::PCAGPUNonRigidFittingPipe()
            :isModelInitialized(false) {}

    PCAGPUNonRigidFittingPipe::~PCAGPUNonRigidFittingPipe() {
        if(isModelInitialized) {
            freeModelCUDA(c_deformModel);
        }
    }

    boost::shared_ptr<PCANonRigidFittingResult>
    PCAGPUNonRigidFittingPipe::_processData(boost::shared_ptr<PCARigidAlignmentSuite> in) {
        /* Load data to cuda device */
        if(!isModelInitialized) {
            auto deformBasis = in->pca_model->getBasisMatrix();
            auto ref = in->pca_model->getReferenceVector();
            auto landmarks = in->pca_model->getLandmarks();
            loadModelToCUDADevice(&this->c_deformModel, deformBasis, ref, landmarks);
            isModelInitialized = true;
        }

        // Filter out non-detected Deformable Model landmarks
        std::vector<int> validLmks = in->pca_model->getLandmarks();
        std::vector<int>::reverse_iterator riter = in->fittingSuite->invalid3dLandmarks.rbegin();
        while (riter != in->fittingSuite->invalid3dLandmarks.rend())
        {
            std::vector<int>::iterator iter_data = validLmks.begin() + *riter;
            iter_data = validLmks.erase(iter_data);
            riter++;
        }

        C_ScanPointCloud c_scanPointCloud;
        loadScanToCUDADevice(&c_scanPointCloud, in->rawCloud, in->fittingSuite->rawCloudLmkIdx,
                             validLmks, in->transformation);

        /* Setup Optimizer */
        auto cost = new PCAGPULandmarkDistanceFunctor<RANK>(this->c_deformModel, c_scanPointCloud);
        ceres::Problem problem;
        double coeff[RANK] = {0,};
        problem.AddResidualBlock(cost, NULL, coeff);
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 1000;
        options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
        options.function_tolerance = 1e-4;

        /* Run Optimization */
        auto summary = ceres::Solver::Summary();
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        std::cout << coeff[0] << std::endl;

        /* Free Resources */
        freeScanCUDA(c_scanPointCloud);

        auto result = boost::make_shared<PCANonRigidFittingResult>();

        result->fitCoeff = Eigen::Map<Eigen::VectorXd>(coeff, RANK).cast<float>();
        std::cout << result->fitCoeff << std::endl;
        result->image = in->image;
        result->fx = in->fx;
        result->fy = in->fy;
        result->transformation = in->transformation;

        return result;
    }
}
