#pragma once

#include <Eigen/Core>

#include "io/pipe.h"
#include "align/rigid.h"
#include "type.h"

namespace telef::align {

    using PCANonRigidFittingResult = struct PCANonRigidFittingResult {
        std::shared_ptr<telef::face::MorphableFaceModel> pca_model;
        Eigen::VectorXf fitCoeff;
        telef::types::ImagePtrT image;
        Eigen::Matrix4f transformation;
        float fx;
        float fy;
    };

    class PCANonRigidFittingPipe : public telef::io::Pipe<PCARigidAlignmentSuite, PCANonRigidFittingResult> {
    private:
        boost::shared_ptr<PCANonRigidFittingResult> _processData(boost::shared_ptr<PCARigidAlignmentSuite> in) override;
    };
}
