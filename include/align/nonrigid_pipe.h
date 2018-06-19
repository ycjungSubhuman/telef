#pragma once

#include <Eigen/Core>

#include "io/pipe.h"
#include "align/rigid.h"
#include "face/cu_model.h"
#include "type.h"

namespace telef::align {

    using PCANonRigidFittingResult = struct PCANonRigidFittingResult {
        std::shared_ptr<telef::face::MorphableFaceModel<RANK>> pca_model;
        Eigen::VectorXf fitCoeff;
        telef::types::ImagePtrT image;
        Eigen::Matrix4f transformation;
        float fx;
        float fy;
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    class PCANonRigidFittingPipe : public telef::io::Pipe<PCARigidAlignmentSuite, PCANonRigidFittingResult> {
    private:
        boost::shared_ptr<PCANonRigidFittingResult> _processData(boost::shared_ptr<PCARigidAlignmentSuite> in) override;
    };

    class PCAGPUNonRigidFittingPipe : public telef::io::Pipe<PCARigidAlignmentSuite, PCANonRigidFittingResult> {
    public:
        PCAGPUNonRigidFittingPipe();
        virtual ~PCAGPUNonRigidFittingPipe();
    private:
        C_PcaDeformModel c_deformModel;
        bool isModelInitialized;
        boost::shared_ptr<PCANonRigidFittingResult> _processData(boost::shared_ptr<PCARigidAlignmentSuite> in) override;
    };
}
