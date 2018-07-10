#pragma once

#include <Eigen/Core>
#include <cublas_v2.h>

#include "io/pipe.h"
#include "align/rigid.h"
#include "face/model_cudahelper.h"
//#include "face/raw_model.h"
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

    class PCAGPUNonRigidFittingPipe : public telef::io::Pipe<PCARigidAlignmentSuite, PCANonRigidFittingResult> {
    public:
        PCAGPUNonRigidFittingPipe();
        PCAGPUNonRigidFittingPipe(const PCAGPUNonRigidFittingPipe &that);
        PCAGPUNonRigidFittingPipe(PCAGPUNonRigidFittingPipe &&that) noexcept;
        PCAGPUNonRigidFittingPipe& operator=(const PCAGPUNonRigidFittingPipe &that);
        PCAGPUNonRigidFittingPipe& operator=(PCAGPUNonRigidFittingPipe &&that);
        virtual ~PCAGPUNonRigidFittingPipe();
    private:
        C_PcaDeformModel c_deformModel;
        cublasHandle_t cublasHandle;
        bool isModelInitialized;
        boost::shared_ptr<PCANonRigidFittingResult> _processData(boost::shared_ptr<PCARigidAlignmentSuite> in) override;
    };
}
