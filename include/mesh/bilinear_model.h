#pragma once

#include <vector>

#include "mesh/mesh.h"

namespace telef::io::dataset {
namespace {
using namespace telef::mesh;
}
/**
 * Bilinear model with N identities and M expressions for each identity
 * Each face have vertex correspondence
 */
class BilinearModel {
public:
  virtual ColorMesh GetMesh(int idIndex, int bsIndex) const = 0;
  virtual ColorMesh GetNeutral(int idIndex) const = 0;
  virtual ColorMesh GetMeanExp(int bsIndex) const = 0;
  virtual int GetVertexCount() const = 0;
  virtual int GetIdCount() const = 0;
  virtual int GetBsCount() const = 0;
};

/**
 * Samples identities and expressions from Bilinaer model
 */
class BilinearModelSampler {
public:
  virtual std::vector<ColorMesh> SampleId(const BilinearModel &bm) = 0;
  virtual std::vector<ColorMesh> SampleEx(const BilinearModel &bm) = 0;
};
} // namespace telef::io::dataset
