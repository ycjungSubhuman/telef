#pragma once

#include "align/nonrigid_pipe.h"
#include "face/model.h"
#include "feature/feature_detector.h"
#include "io/normaldepth_pipe.h"
#include "io/pipe.h"

namespace {
using namespace telef::align;
}

namespace telef::face {
class MorphableModelFeederPipe
    : public telef::io::
          Pipe<telef::feature::FittingSuite, PCANonRigidAlignmentSuite> {
private:
  using MModelTptr = std::shared_ptr<telef::face::MorphableFaceModel>;
  using BaseT =
      telef::io::Pipe<telef::feature::FittingSuite, PCANonRigidAlignmentSuite>;
  using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

  MModelTptr pca_model;

  boost::shared_ptr<PCANonRigidAlignmentSuite>
  _processData(boost::shared_ptr<telef::feature::FittingSuite> in) override;

public:
  explicit MorphableModelFeederPipe(MModelTptr model);
};

class MultipleModelFeederPipe
    : public telef::io::
          Pipe<telef::feature::FittingSuite, PCANonRigidAlignmentSuite> {
private:
  using MModelTptr = std::shared_ptr<telef::face::MorphableFaceModel>;
  using BaseT =
      telef::io::Pipe<telef::feature::FittingSuite, PCANonRigidAlignmentSuite>;
  using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

  MModelTptr pca_model_selected;
  std::vector<MModelTptr> pca_models;
  float reg;
  telef::io::MeshNormalDepthRenderer render;

  boost::shared_ptr<PCANonRigidAlignmentSuite>
  _processData(boost::shared_ptr<telef::feature::FittingSuite> in) override;

public:
  explicit MultipleModelFeederPipe(const std::vector<MModelTptr> &model, telef::io::MeshNormalDepthRenderer render, float reg=0.0001);
};
} // namespace telef::face
