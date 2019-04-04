#pragma once
#include "mesh/bilinear_model.h"
#include <experimental/filesystem>
#include <vector>

namespace {
using namespace telef::mesh;
namespace fs = std::experimental::filesystem;
} // namespace

namespace telef::io::dataset {

class FaceWarehouse : public BilinearModel {
public:
  FaceWarehouse(fs::path root);

  ColorMesh GetMesh(int idIndex, int bsIndex) const override;
  ColorMesh GetNeutral(int idIndex) const override;
  ColorMesh GetMeanExp(int bsIndex) const override;
  int GetVertexCount() const override;
  int GetIdCount() const override;
  int GetBsCount() const override;

private:
  fs::path m_root;
  Eigen::VectorXi m_faces;
  std::vector<Eigen::VectorXf> m_mean_exps;
  std::vector<std::vector<Eigen::VectorXf>> m_positions;
  int m_idCount;
  int m_bsCount;
};

class FaceWarehouseAllSampler : public BilinearModelSampler {
public:
  std::vector<ColorMesh> SampleId(const BilinearModel &bm) override;
  std::vector<ColorMesh> SampleEx(const BilinearModel &bm) override;
};

} // namespace telef::io::dataset
