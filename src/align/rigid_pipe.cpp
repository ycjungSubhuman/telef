#include <iostream>

#include <pcl/registration/transformation_estimation_svd_scale.h>

#include "align/rigid_pipe.h"
#include "face/model.h"
#include "util/eigen_pcl.h"

using namespace std;
using namespace telef::feature;
using namespace telef::types;
using namespace telef::face;

namespace telef::align {
PCARigidFittingPipe::PCARigidFittingPipe(MModelTptr model)
    : BaseT(), pca_model(model),
      transformation(Eigen::Matrix4f::Identity(4, 4)) {
  // Generate Mean Face Template
  meanMesh = pca_model->genMesh(
      Eigen::VectorXf::Zero(pca_model->getShapeRank()),
      Eigen::VectorXf::Zero(pca_model->getExpressionRank()));

  // Save initial point cloud for rigid fitting
  initShape = telef::util::convert(meanMesh.position);
}

boost::shared_ptr<PCANonRigidAlignmentSuite>
PCARigidFittingPipe::_processData(boost::shared_ptr<FittingSuite> in) {
  std::vector<int> pca_lmks = pca_model->getLandmarks();
  auto in_lmks = in->landmark3d;

  std::vector<int> corr_tgt(in_lmks->points.size());
  std::iota(
      std::begin(corr_tgt),
      std::end(corr_tgt),
      0); // Fill with range 0, ..., n.

  // Check if we detected good landmarks
  if (corr_tgt.size() + in->invalid3dLandmarks.size() == pca_lmks.size()) {
    // Remove invalid Correspondences
    std::vector<int>::reverse_iterator riter = in->invalid3dLandmarks.rbegin();
    while (riter != in->invalid3dLandmarks.rend()) {
      std::vector<int>::iterator iter_data = pca_lmks.begin() + *riter;
      iter_data = pca_lmks.erase(iter_data);
      riter++;
    }

    Eigen::Matrix4f currentTransform;
    pcl::registration::
        TransformationEstimationSVDScale<pcl::PointXYZ, pcl::PointXYZRGBA>
            svd;
    svd.estimateRigidTransformation(
        *initShape, pca_lmks, *in_lmks, corr_tgt, currentTransform);
    this->transformation = currentTransform;
  } else {
    // std::cout << "\n Didn't detect all landmarks, using last transformation:"
    // << std::endl;
  }

  // std::cout << "\n Transformtion Matrix: \n" << this->transformation <<
  // std::endl;
  auto alignment = boost::shared_ptr<PCANonRigidAlignmentSuite>(
      new PCANonRigidAlignmentSuite());
  alignment->fittingSuite = in;
  alignment->pca_model = pca_model;
  alignment->transformation = this->transformation;
  alignment->image = in->rawImage;
  alignment->fx = in->fx;
  alignment->fy = in->fy;
  alignment->rawCloud = in->rawCloud;
  return alignment;
}
} // namespace telef::align
