#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>
#include <cmath>

#include "align/nonrigid_pipe.h"
#include "type.h"

#define EPS 0.005

using namespace telef::types;
using namespace telef::face;

namespace {
    Eigen::VectorXf getNearestPoints(const Eigen::VectorXf &meshPoints, CloudConstPtrT scanCloud, int &numValidPoints) {
        auto posCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        posCloud->resize(scanCloud->size());
        for(unsigned long i=0; i<scanCloud->points.size(); i++) {
            posCloud->points[i].x = scanCloud->points[i].x;
            posCloud->points[i].y = scanCloud->points[i].y;
            posCloud->points[i].z = scanCloud->points[i].z;
        }

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(posCloud);

        Eigen::VectorXf result = meshPoints;

        numValidPoints = 0;
        for(unsigned long i=0; i<meshPoints.size()/3; i++) {
            auto point = pcl::PointXYZ(meshPoints[i], meshPoints[i+1], meshPoints[i+2]);
            auto resultIndices = std::vector<int>(1);
            auto resultDists = std::vector<float>(1);
            auto numPointsFound = kdtree.nearestKSearch(point, 1, resultIndices, resultDists);

            if(numPointsFound == 1 && resultDists[0] < EPS) { // Valid Point
                auto corrPoint = posCloud->points[resultIndices[0]];
                result[3*i] = corrPoint.x;
                result[3*i+1] = corrPoint.y;
                result[3*i+2] = corrPoint.z;
                numValidPoints++;
            }
        }
        std::cout << "Points Found: " << numValidPoints << std::endl;
        if(numValidPoints == 0) numValidPoints = 1;
        return result;
    }

    template<int CoeffRank>
    class FaceCostFunction : public ceres::SizedCostFunction<1, CoeffRank> {
    private:
        std::vector<int> meshLandmark3d;
        std::vector<int> badPointInds;
        CloudConstPtrT scanLandmark3d;

        CloudConstPtrT scanPc;
        std::shared_ptr<MorphableFaceModel<CoeffRank>> model;
        Eigen::Matrix4f transformation;

        float landmarkCoeff;
        float nearestCoeff;
        float regularizerCoeff;

        Eigen::VectorXf lmkInd2Pos(const Eigen::VectorXf &fullMeshPos) const {
            Eigen::VectorXf result(meshLandmark3d.size()*3);

            for(unsigned long i=0; i<meshLandmark3d.size(); i++) {
                result[3*i] = fullMeshPos[3*meshLandmark3d[i]];
                result[3*i+1] = fullMeshPos[3*meshLandmark3d[i]+1];
                result[3*i+2] = fullMeshPos[3*meshLandmark3d[i]+2];
            }

            return result;
        }

        void getLandmarkRes(const Eigen::VectorXf &meshPos, const double * const params, double *residual,
                            double *jacobian, bool isJacobianRequired) const {
            Eigen::VectorXf meshLmk3d = lmkInd2Pos(meshPos);

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
                    Eigen::Vector3f ptSubt;
                    ptSubt <<
                            meshLmk3d[3 * i] - scanLandmark3d->points[validPointCount].x,
                            meshLmk3d[3 * i + 1] - scanLandmark3d->points[validPointCount].y,
                            meshLmk3d[3 * i + 2] - scanLandmark3d->points[validPointCount].z;

                    *residual += landmarkCoeff * ptSubt.squaredNorm();
                    if(isJacobianRequired) {
                        for (unsigned long j = 0; j < CoeffRank; j++) {
                            Eigen::Vector3f basis;
                            auto modelBasis = model->getBasis(j);
                            basis << modelBasis[3 * meshLandmark3d[i]],
                                    modelBasis[3 * meshLandmark3d[i] + 1],
                                    modelBasis[3 * meshLandmark3d[i] + 2];

                            jacobian[j] += 2 * landmarkCoeff * (ptSubt.array() * basis.array()).sum();

                        }
                    }
                    validPointCount++;
                }
            }

            if(validPointCount == 0) validPointCount = 1;

            *residual /= validPointCount;

            if(isJacobianRequired) {
                for (unsigned long j = 0; j < CoeffRank; j++) {
                    jacobian[j] /= validPointCount;
                }
            }
        }

        void getNearestRes(const Eigen::VectorXf &meshPos, const double * const params, double *residual,
                           double *jacobian, bool isJacobianRequired) const {
            int numValidPoint;
            auto nearestPts = getNearestPoints(meshPos, scanPc, numValidPoint);
            *residual = nearestCoeff * (meshPos - nearestPts).squaredNorm()/numValidPoint;
            if (isJacobianRequired) {
                for (unsigned long j = 0; j < CoeffRank; j++) {
                    jacobian[j] =
                            2 * nearestCoeff * ((meshPos - nearestPts).array() * model->getBasis(j).array()).sum() / numValidPoint;
                }
            }
        }

        void getRegularizerRes(const Eigen::VectorXf &meshPos, const double * const params, double *residual,
                               double *jacobian, bool isJacobianRequired) const {
            double res = 0.0;
            double diff = 0.0;
            for (int i=0; i<CoeffRank; i++) {
                res += regularizerCoeff * std::pow(params[i], 2.0);
                if(isJacobianRequired) {
                    jacobian[i] = 2 * regularizerCoeff * params[i];
                }
            }
            *residual = res;
        }

    public:
        FaceCostFunction(
                std::vector<int> meshLandmark3d, CloudConstPtrT scanLandmark3d,
                CloudConstPtrT scanPc, std::shared_ptr<MorphableFaceModel<CoeffRank>> model,
                std::vector<int> badLmkInds, Eigen::Matrix4f transformation,
                float landmarkCoeff, float nearestCoeff, float regularizerCoeff) :
        meshLandmark3d(meshLandmark3d),
        scanLandmark3d(scanLandmark3d),
        scanPc(scanPc),
        model(model),
        badPointInds(badLmkInds),
        transformation(transformation),
        landmarkCoeff(landmarkCoeff),
        nearestCoeff(nearestCoeff),
        regularizerCoeff(regularizerCoeff) {
            std::cout << meshLandmark3d.size() << std::endl;
            std::cout << scanLandmark3d->points.size() << std::endl;
            std::cout << badPointInds.size() << std::endl;

            assert(meshLandmark3d.size() == scanLandmark3d->points.size() + badPointInds.size());
        }

        virtual ~FaceCostFunction() {}
        bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const override {
            std::cout << parameters[0][0] << std::endl;
            double lmkRes = 0;
            double nearRes = 0;
            double regRes = 0;
            double lmkJ[CoeffRank] = {0,};
            double nearJ[CoeffRank] = {0,};
            double regJ[CoeffRank] = {0,};
            bool isJacobianRequired = jacobians != nullptr && jacobians[0] != nullptr;

            auto m = model->genMesh(parameters[0], CoeffRank);
            m.applyTransform(transformation);
            auto meshPos = m.position;

            getLandmarkRes(meshPos, parameters[0], &lmkRes, lmkJ, isJacobianRequired);
            getNearestRes(meshPos, parameters[0], &nearRes, nearJ, isJacobianRequired);
            getRegularizerRes(meshPos, parameters[0], &regRes, regJ, isJacobianRequired);

            residuals[0] = lmkRes + nearRes + regRes;
            if(isJacobianRequired) {
                for (int i = 0; i < CoeffRank; i++) {
                    jacobians[0][i] = lmkJ[i] + nearJ[i] + regJ[i];
                }
            }

            return true;
        }
    };
}

namespace telef::align {
    boost::shared_ptr<PCANonRigidFittingResult>
    PCANonRigidFittingPipe::_processData(boost::shared_ptr<telef::align::PCARigidAlignmentSuite> in)
    {
        auto result = boost::make_shared<PCANonRigidFittingResult>();
        result->pca_model = in->pca_model;

        ceres::Problem problem;
        auto cost = new FaceCostFunction<150>(in->pca_model->getLandmarks(), in->fittingSuite->landmark3d, in->rawCloud,
                                          in->pca_model, in->fittingSuite->invalid3dLandmarks, in->transformation,
                                          100.0, 80.0, 0.000002);
        double coeff[150] = {0,};
        problem.AddResidualBlock(cost, nullptr, coeff);
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.use_nonmonotonic_steps = true;
        options.function_tolerance = 1e-10;
        auto summary = ceres::Solver::Summary();
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        std::cout << coeff[0] << std::endl;
        std::cout << "wat??" << std::endl;

        result->fitCoeff = Eigen::Map<Eigen::VectorXd>(coeff, 150).cast<float>();
        std::cout << result->fitCoeff << std::endl;
        result->image = in->image;
        result->fx = in->fx;
        result->fy = in->fy;
        result->transformation = in->transformation;

        return result;
    }
}