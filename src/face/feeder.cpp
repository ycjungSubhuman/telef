#include "face/feeder.h"

namespace telef::face {

boost::shared_ptr<PCANonRigidAlignmentSuite>
MorphableModelFeederPipe::_processData(
    boost::shared_ptr<telef::feature::FittingSuite> in) {
  auto result = boost::make_shared<PCANonRigidAlignmentSuite>();
  result->fittingSuite = in;
  result->pca_model = pca_model;
  result->transformation = Eigen::Matrix4f::Identity();
  result->image = in->rawImage;
  result->fx = in->fx;
  result->fy = in->fy;
  result->shapeCoeff = Eigen::VectorXf::Zero(pca_model->getShapeRank());
  result->expressionCoeff =
    Eigen::VectorXf::Zero(pca_model->getExpressionRank());
  result->rawCloud = in->rawCloud;
  return result;
}

MorphableModelFeederPipe::MorphableModelFeederPipe(
    MorphableModelFeederPipe::MModelTptr model)
    : BaseT(), pca_model(model) {}
} // namespace telef::face
