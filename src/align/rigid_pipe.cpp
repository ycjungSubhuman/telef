#include <Eigen/Geometry>
#include <iostream>

#include "align/rigid_pipe.h"
#include "face/model.h"
#include "util/eigen_pcl.h"

using namespace std;
using namespace telef::feature;
using namespace telef::types;
using namespace telef::face;

namespace telef::align {

boost::shared_ptr<PCANonRigidAlignmentSuite> PCARigidFittingPipe::_processData(
    boost::shared_ptr<PCANonRigidAlignmentSuite> in) {
  std::vector<int> pca_lmks = in->pca_model->getLandmarks();
  auto in_lmks = in->fittingSuite->landmark3d;

  Eigen::VectorXf ref = in->pca_model->getReferenceVector();
  Eigen::Matrix3Xf mesh_pts_t =
      Eigen::Map<Eigen::Matrix3Xf>(ref.data(), 3, ref.size() / 3);
  Eigen::MatrixXf mesh_lmk_pts(in_lmks->size(), 3);

  for (int i = 0; i < in_lmks->size(); i++) {
    mesh_lmk_pts.row(i) = mesh_pts_t.col(pca_lmks[i]);
  }

  Eigen::MatrixXf lmk_pts(in_lmks->size(), 3);
  for (int i = 0; i < in_lmks->size(); i++) {
    lmk_pts(i, 0) = in_lmks->points[i].x;
    lmk_pts(i, 1) = in_lmks->points[i].y;
    lmk_pts(i, 2) = in_lmks->points[i].z;
  }

  Eigen::MatrixXf transformation =
      Eigen::umeyama(mesh_lmk_pts.transpose(), lmk_pts.transpose());

  in->transformation = transformation;
  return in;
}
} // namespace telef::align
