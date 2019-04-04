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

  Eigen::VectorXf ref =
    in->pca_model->genPosition(in->shapeCoeff, in->expressionCoeff);
  Eigen::Matrix3Xf mesh_pts_t =
      Eigen::Map<Eigen::Matrix3Xf>(ref.data(), 3, ref.size() / 3);
  std::vector<int> selection = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67,
};
  Eigen::MatrixXf mesh_lmk_pts(selection.size(), 3);

  for (int i = 0; i < selection.size(); i++) {
    mesh_lmk_pts.row(i) = mesh_pts_t.col(pca_lmks[selection[i]]);
  }

  Eigen::MatrixXf lmk_pts(selection.size(), 3);
  for (int i = 0; i < selection.size(); i++) {
    lmk_pts(i, 0) = in_lmks->points[selection[i]].x;
    lmk_pts(i, 1) = in_lmks->points[selection[i]].y;
    lmk_pts(i, 2) = in_lmks->points[selection[i]].z;
  }

  Eigen::MatrixXf transformation =
      Eigen::umeyama(mesh_lmk_pts.transpose(), lmk_pts.transpose());

  in->transformation = transformation;
  return in;
}
} // namespace telef::align
