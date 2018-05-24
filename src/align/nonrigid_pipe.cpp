#include <boost/make_shared.hpp>
#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <Eigen/Core>
#include "ceres/autodiff_cost_function.h"
#include <cmath>

#include "align/nonrigid_pipe.h"
#include "type.h"

#define EPS 0.005

using namespace telef::types;
using namespace telef::face;

namespace {
    template<int CoeffRank>
    class PCALandmarkDistanceFunctor {
    public:
        explicit PCALandmarkDistanceFunctor(std::vector<int> meshLandmark3d, CloudConstPtrT scanLandmark3d,
                                            std::shared_ptr<MorphableFaceModel<CoeffRank>> model,
                                            std::vector<int> badLmkInds, Eigen::Matrix4f transformation,
                                            double landmarkWeight) :
        meshLandmark3d(meshLandmark3d),
        scanLandmark3d(scanLandmark3d),
        model(model),
        badPointInds(badLmkInds),
        transformation(transformation),
        landmarkCoeff(landmarkWeight) {
            std::cout << meshLandmark3d.size() << std::endl;
            std::cout << scanLandmark3d->points.size() << std::endl;
            std::cout << badPointInds.size() << std::endl;

            assert(meshLandmark3d.size() == scanLandmark3d->points.size() + badPointInds.size());
        }

        template <typename T>
        bool operator()(const T* pcaCoeff, T* residuals) const {
            residuals[0] = T(0.0);

            auto m = model->genPositionCeres(pcaCoeff, CoeffRank);
            auto meshPos = applyTransform(m, transformation);

            //Eigen::VectorXf
            Eigen::Matrix<T, Eigen::Dynamic, 1> meshLmk3d = lmkInd2Pos(meshPos);

            int validPointCount = 0;

            for(unsigned long i=0; i<meshLandmark3d.size(); i++) {
                bool isPointValid = true;
                for(unsigned long j=0; j<badPointInds.size(); j++) {
                    if(i==badPointInds[j]) {
                        isPointValid = false;
                        break;
                    }
                }

                if(isPointValid) {
                    Eigen::Matrix<T, 3, 1> ptSubt;
                    ptSubt <<
                           meshLmk3d[3 * i] - T(scanLandmark3d->points[validPointCount].x),
                            meshLmk3d[3 * i + 1] - T(scanLandmark3d->points[validPointCount].y),
                            meshLmk3d[3 * i + 2] - T(scanLandmark3d->points[validPointCount].z);

                    auto normDist = ptSubt.norm();
                    residuals[0] = residuals[0] + (normDist);
                    validPointCount++;
                }
            }

            if(validPointCount == 0) validPointCount = 1;

            // Cost is Average error
            residuals[0] = residuals[0] / T(validPointCount);
            return true;
        }


        static ceres::CostFunction* create(std::vector<int> meshLandmark3d, CloudConstPtrT scanLandmark3d,
                                           std::shared_ptr<MorphableFaceModel<CoeffRank>> model,
                                           std::vector<int> badLmkInds, Eigen::Matrix4f transformation,
                                           float landmarkWeight) {
            return new ceres::AutoDiffCostFunction<PCALandmarkDistanceFunctor, /*ResDim*/ 1, /*Param1Dim*/ CoeffRank>(
                    new PCALandmarkDistanceFunctor<CoeffRank>(meshLandmark3d,scanLandmark3d,model,
                                                              badLmkInds,transformation,landmarkWeight));
        }

    private:
        std::vector<int> meshLandmark3d;
        std::vector<int> badPointInds;
        CloudConstPtrT scanLandmark3d;

        std::shared_ptr<MorphableFaceModel<CoeffRank>> model;
        Eigen::Matrix4f transformation;

        double landmarkCoeff;

        /** Convert full mesh position to landmark poisitions */
        template <class T>
        Eigen::Matrix<T, Eigen::Dynamic, 1> lmkInd2Pos(const Eigen::Matrix<T, Eigen::Dynamic, 1> &fullMeshPos) const {
            Eigen::Matrix<T, Eigen::Dynamic, 1> result(meshLandmark3d.size()*3);

            for(unsigned long i=0; i<meshLandmark3d.size(); i++) {
                result[3*i] = fullMeshPos[3*meshLandmark3d[i]];
                result[3*i+1] = fullMeshPos[3*meshLandmark3d[i]+1];
                result[3*i+2] = fullMeshPos[3*meshLandmark3d[i]+2];
            }

            return result;
        }

        template <typename T>
        Eigen::Matrix<T, Eigen::Dynamic, 1> applyTransform(Eigen::Matrix<T, Eigen::Dynamic, 1> &meshPos,
                                                           Eigen::Matrix4f transform) const
        {
            Eigen::Map<Eigen::Matrix<T, 3, Eigen::Dynamic>> v(meshPos.data(), 3, meshPos.size()/3);
            Eigen::Matrix<T, 3, Eigen::Dynamic> result = (transform.cast<T>() * v.colwise().homogeneous()).colwise().hnormalized();
            return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>{result.data(), result.size()};
        }
    };
}

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
        // The ownership of 'cost' is moved to 'probelm'. So we don't delete cost outsideof 'problem'.
        problem.AddResidualBlock(cost, loss_function, coeff);
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 1000;
	//options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
        options.function_tolerance = 1e-3;
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
}
