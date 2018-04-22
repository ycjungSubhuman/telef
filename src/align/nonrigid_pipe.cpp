#include "align/nonrigid_pipe.h"

namespace telef::align {
    boost::shared_ptr<PCANonRigidFittingResult>
    PCANonRigidFittingPipe::_processData(boost::shared_ptr<telef::align::PCARigidAlignmentSuite> in)
    {
        auto result = boost::make_shared<PCANonRigidFittingResult>();
        result->pca_model = in->pca_model;
        // TODO : Implement Actural non-rigid fitting
        result->fitCoeff = Eigen::VectorXf::Zero(in->pca_model->getRank());
        result->image = in->image;
        result->fx = in->fx;
        result->fy = in->fy;

        return result;
    }
}