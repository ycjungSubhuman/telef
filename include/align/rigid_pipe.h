#pragma once

#include "align/nonrigid_pipe.h"
#include "face/model.h"
#include "io/pipe.h"
#include "mesh/mesh.h"

namespace telef::align {
/**
 * Rigid alignment of PCA Template to FittingSuite data
 */
// template <int ShapeRank>
class PCARigidFittingPipe : public telef::io::Pipe<telef::feature::FittingSuite,
                                                   PCANonRigidAlignmentSuite> {
private:
  using MModelTptr = std::shared_ptr<telef::face::MorphableFaceModel>;
  using BaseT =
      telef::io::Pipe<telef::feature::FittingSuite, PCANonRigidAlignmentSuite>;
  using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

  // Odd error =operator is deleted
  MModelTptr pca_model;
  telef::mesh::ColorMesh meanMesh;
  PtCldPtr initShape;
  Eigen::Matrix4f transformation;
  // TODO: Keep last frame transformation matrix (Trans1 * Trans2)
  // or pointcloud in to optimize between frames?

  boost::shared_ptr<PCANonRigidAlignmentSuite>
  _processData(boost::shared_ptr<telef::feature::FittingSuite> in) override;

public:
  PCARigidFittingPipe(MModelTptr model);
};

class LmkToScanRigidFittingPipe
    : public telef::io::Pipe<telef::feature::FeatureDetectSuite,
                             telef::feature::FittingSuite> {
private:
  using BaseT = telef::io::Pipe<telef::feature::FeatureDetectSuite,
                                telef::feature::FittingSuite>;
  // using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

  Eigen::Matrix4f transformation;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr landmark3d;

  boost::shared_ptr<telef::feature::FittingSuite>
  _processData(telef::feature::FeatureDetectSuite::Ptr in) override;

public:
  LmkToScanRigidFittingPipe();
};

Eigen::VectorXf alignLandmark(telef::types::CloudConstPtrT scanCloud,
                              telef::types::CloudConstPtrT landmarkCloud,
                              Eigen::VectorXf meshPosition,
                              std::vector<int> landmarkIndices);
} // namespace telef::align
