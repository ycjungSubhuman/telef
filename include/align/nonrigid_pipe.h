#pragma once

#include <Eigen/Core>
#include <cublas_v2.h>

#include "io/pipe.h"
#include "face/model.h"
#include "face/model_cudahelper.h"
#include "feature/feature_detector.h"
#include "type.h"

namespace telef::align {
    // Data needed for fitting face with alignment
    using PCANonRigidAlignmentSuite =
    struct PCANonRigidAlignmentSuite {
        using Ptr = std::shared_ptr<PCANonRigidAlignmentSuite>;
        using ConstPtr = std::shared_ptr<const PCANonRigidAlignmentSuite>;

        boost::shared_ptr<telef::feature::FittingSuite> fittingSuite;

        std::shared_ptr<telef::face::MorphableFaceModel> pca_model;
        // Rigid alignment
        Eigen::Matrix4f transformation;
        telef::types::ImagePtrT image;
        CloudConstPtrT rawCloud;

        float fx;
        float fy;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    using PCANonRigidFittingResult = struct PCANonRigidFittingResult {
        std::shared_ptr<telef::face::MorphableFaceModel> pca_model;
        Eigen::VectorXf shapeCoeff;
        Eigen::VectorXf expressionCoeff;
        telef::types::ImagePtrT image;
        telef::types::CloudConstPtrT cloud;
        Eigen::Matrix4f transformation;
        CloudConstPtrT landmark3d;
        float fx;
        float fy;
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    class PCAGPUNonRigidFittingPipe : public telef::io::Pipe<PCANonRigidAlignmentSuite, PCANonRigidFittingResult> {
    public:
        PCAGPUNonRigidFittingPipe();
        PCAGPUNonRigidFittingPipe(const float geoWeight, const int geoMaxPoints,
                                  const float geoSearchRadius, const bool addGeoTerm=true);
        PCAGPUNonRigidFittingPipe(const PCAGPUNonRigidFittingPipe &that);
        PCAGPUNonRigidFittingPipe(PCAGPUNonRigidFittingPipe &&that) noexcept;
        PCAGPUNonRigidFittingPipe& operator=(const PCAGPUNonRigidFittingPipe &that);
        PCAGPUNonRigidFittingPipe& operator=(PCAGPUNonRigidFittingPipe &&that);
        virtual ~PCAGPUNonRigidFittingPipe();
    private:
        C_PcaDeformModel c_deformModel;
        cublasHandle_t cublasHandle;
        bool isModelInitialized;
        boost::shared_ptr<PCANonRigidFittingResult> _processData(boost::shared_ptr<PCANonRigidAlignmentSuite> in) override;

        static const std::vector<int> landmarkSelection;

        // Geometric Term
        float geoWeight;
        int geoMaxPoints;
        float geoSearchRadius;
        bool addGeoTerm;
    };
}
