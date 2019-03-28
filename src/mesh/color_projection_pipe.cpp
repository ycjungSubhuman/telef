#include "mesh/color_projection_pipe.h"
#include "mesh/colormapping.h"
#include <boost/make_shared.hpp>

using namespace telef::align;

namespace telef::mesh {
boost::shared_ptr<ProjectionSuite> Fitting2ProjectionPipe::_processData(
    boost::shared_ptr<PCANonRigidFittingResult> in) {
  auto result = boost::make_shared<ProjectionSuite>();
  result->image = in->image;
  result->pca_model = in->pca_model;
  result->fitResult =
      in->pca_model->genMesh(in->shapeCoeff, in->expressionCoeff);
  result->fitResult.applyTransform(in->transformation);
  result->fx = in->fx;
  result->fy = in->fy;

  return result;
}

boost::shared_ptr<telef::mesh::ColorMesh>
ColorProjectionPipe::_processData(boost::shared_ptr<ProjectionSuite> in) {
  projectColor(in->image, in->fitResult, in->fx, in->fy);
  return boost::make_shared<telef::mesh::ColorMesh>(in->fitResult);
}
} // namespace telef::mesh
