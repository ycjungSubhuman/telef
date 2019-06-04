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

PCARigidFittingPipe::PCARigidFittingPipe() :
    m_prev_scale(0.0f)
{}

boost::shared_ptr<PCANonRigidAlignmentSuite> PCARigidFittingPipe::_processData(
    boost::shared_ptr<PCANonRigidAlignmentSuite> in) {
  std::vector<int> pca_lmks = in->pca_model->getLandmarks();
  auto in_lmks = in->fittingSuite->landmark3d;

  Eigen::VectorXf ref =
    in->pca_model->genPosition(in->shapeCoeff, in->expressionCoeff);
  Eigen::Matrix3Xf mesh_pts_t =
      Eigen::Map<Eigen::Matrix3Xf>(ref.data(), 3, ref.size() / 3);
  std::vector<int> selection = {
                                0, 1, 2, 3,
                                13, 14, 15, 16,
                                27, 28, 29, 30,
                                33, 36, 39, 42, 45};
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

  Eigen::MatrixXf transformation;
  if(m_prev_scale == 0.0f)
    {
      transformation = Eigen::umeyama(mesh_lmk_pts.transpose(), lmk_pts.transpose());
      m_prev_scale = transformation.block(0,0,3,0).norm();
    }
  else
    {
      transformation = Eigen::umeyama(
          m_prev_scale*mesh_lmk_pts.transpose(), lmk_pts.transpose(), false);
      transformation.block(0,0,3,3) *= m_prev_scale;
    }

  in->transformation = transformation;
  return in;
}
} // namespace telef::align
