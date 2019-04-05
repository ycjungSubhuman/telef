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
class PCARigidFittingPipe
    : public telef::io::
          Pipe<PCANonRigidAlignmentSuite, PCANonRigidAlignmentSuite> {
public:
  PCARigidFittingPipe();
private:
  using MModelTptr = std::shared_ptr<telef::face::MorphableFaceModel>;
  using BaseT =
      telef::io::Pipe<PCANonRigidAlignmentSuite, PCANonRigidAlignmentSuite>;
  using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;


  float m_prev_scale;

  boost::shared_ptr<PCANonRigidAlignmentSuite>
  _processData(boost::shared_ptr<PCANonRigidAlignmentSuite> in) override;
};

class LmkToScanRigidFittingPipe : public telef::io::Pipe<
                                      telef::feature::FeatureDetectSuite,
                                      telef::feature::FittingSuite> {
private:
  using BaseT = telef::io::
      Pipe<telef::feature::FeatureDetectSuite, telef::feature::FittingSuite>;
  // using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

  Eigen::Matrix4f transformation;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr landmark3d;

  boost::shared_ptr<telef::feature::FittingSuite>
  _processData(telef::feature::FeatureDetectSuite::Ptr in) override;

public:
  LmkToScanRigidFittingPipe();
};

Eigen::VectorXf alignLandmark(
    telef::types::CloudConstPtrT scanCloud,
    telef::types::CloudConstPtrT landmarkCloud,
    Eigen::VectorXf meshPosition,
    std::vector<int> landmarkIndices);
} // namespace telef::align
