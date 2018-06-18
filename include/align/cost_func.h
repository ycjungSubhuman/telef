#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "face/model.h"
#include "type.h"

namespace {
    using namespace telef::face;
    using namespace telef::types;
}

namespace telef::align{
    template<int CoeffRank>
    class PCALandmarkDistanceFunctor {
    public:
        explicit PCALandmarkDistanceFunctor(std::vector<int> meshLandmark3d, CloudConstPtrT scanLandmark3d,
                                            std::shared_ptr<MorphableFaceModel<CoeffRank>> model,
                                            std::vector<int> badLmkInds, Eigen::Matrix4f &transformation,
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
        bool operator()(const T* pcaCoeff, const T* rigidCoeff, T* residuals) const {
            residuals[0] = T(0.0);

            auto m = model->genPositionCeres(pcaCoeff, CoeffRank);
	    Eigen::Matrix<T, 4, 4> rigidFineTune = buildTransform(rigidCoeff);
	    Eigen::Matrix<T, 4, 4> totalTransform = rigidFineTune * transformation.cast<T>();
            auto meshPos = applyTransform<T>(m, totalTransform);

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

	    // Calculate L2 norm
	    T l2norm = T(0.0);
	    
	    for (int i=0; i<CoeffRank; i++) {
		l2norm += pcaCoeff[i] * pcaCoeff[i];
	    }

            // Cost is Average error
            residuals[0] = residuals[0] / T(validPointCount) + T(0.01) * l2norm;
            return true;
        }


        static ceres::CostFunction* create(std::vector<int> meshLandmark3d, CloudConstPtrT scanLandmark3d,
                                           std::shared_ptr<MorphableFaceModel<CoeffRank>> model,
                                           std::vector<int> badLmkInds, Eigen::Matrix4f &transformation,
                                           float landmarkWeight) {
            return new ceres::AutoDiffCostFunction<PCALandmarkDistanceFunctor, /*ResDim*/ 1, /*Param1Dim*/ CoeffRank, /*Param2Dim*/ 7>(
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
	//T yaw, T pitch, T roll, T tx, T ty, T tz, T scale
	Eigen::Matrix<T, 4, 4> buildTransform(const T *rigidCoeff) const {
	    T yaw = rigidCoeff[0];
	    T pitch = rigidCoeff[1];
	    T roll = rigidCoeff[2];
	    T tx = rigidCoeff[3];
	    T ty = rigidCoeff[4];
	    T tz = rigidCoeff[5];
	    T scale = rigidCoeff[6];

	    Eigen::Matrix<T, 3, 1> xAxis;
	    Eigen::Matrix<T, 3, 1> yAxis;
	    Eigen::Matrix<T, 3, 1> zAxis;
	    xAxis << T(1.0f), T(0.0f), T(0.0f);
	    yAxis << T(0.0f), T(1.0f), T(0.0f);
	    zAxis << T(0.0f), T(0.0f), T(1.0f);

	    Eigen::Transform<T, 3, Eigen::Affine> t( Eigen::AngleAxis<T>(yaw, xAxis)
				    * Eigen::AngleAxis<T>(pitch, yAxis)
				    * Eigen::AngleAxis<T>(roll, zAxis) );
	    
	    Eigen::Matrix<T, 3, 1> translateVector;
	    translateVector << tx, ty, tz;
	    return t.translate(translateVector).scale(scale).matrix();
	}

        template <typename T>
        Eigen::Matrix<T, Eigen::Dynamic, 1> applyTransform(Eigen::Matrix<T, Eigen::Dynamic, 1> &meshPos,
                                                           Eigen::Matrix<T, 4, 4> &transform) const
        {
            Eigen::Map<Eigen::Matrix<T, 3, Eigen::Dynamic>> v(meshPos.data(), 3, meshPos.size()/3);
            Eigen::Matrix<T, 3, Eigen::Dynamic> result = (transform * v.colwise().homogeneous()).colwise().hnormalized();
            return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>{result.data(), result.size()};
        }
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    };
}
