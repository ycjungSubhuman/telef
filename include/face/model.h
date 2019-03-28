#pragma once

#include <algorithm>
#include <ctime>
#include <exception>
#include <experimental/filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "face/deformation_model.h"
#include "io/landmark.h"
#include "io/matrix.h"
#include "io/ply/meshio.h"
#include "mesh/mesh.h"
#include "type.h"
#include "util/eigen_pcl.h"
#include <Eigen/Dense>
#include <pcl/registration/transformation_estimation_svd_scale.h>

namespace {
namespace fs = std::experimental::filesystem;
using namespace telef::mesh;
using namespace telef::io;
} // namespace

namespace telef::face {

/** PCA face model using PCA model of deformation btw reference mesh and
 * samples*/
class MorphableFaceModel {
private:
  std::shared_ptr<LinearModel> shapeModel;
  std::shared_ptr<LinearModel> expressionModel;
  ColorMesh refMesh;
  std::vector<int> landmarks;
  std::random_device rd;
  std::mt19937 mt;

public:
  /** Construct PCA Model using a list of mesh files */
  MorphableFaceModel(fs::path refSamplePath,
                     const std::vector<fs::path> &shapeSamplePaths,
                     const std::vector<fs::path> &expressionSamplePaths,
                     fs::path landmarkIdxPath, int shapeRank,
                     int expressionRank);

  /** Load from existing model file */
  explicit MorphableFaceModel(fs::path fileName);

  /** Save this model to a file */
  void save(fs::path fileName);

  /** Generate a xyzxyz... position vector using given coefficients */
  Eigen::VectorXf genPosition(Eigen::VectorXf shapeCoeff,
                              Eigen::VectorXf expressionCoeff);

  Eigen::VectorXf genPosition(const double *shapeCoeff, int shapeCoeffSize,
                              const double *expressionCoeff,
                              int expressionCoeffSize);

  /** Generate a ColorMesh using given coefficients */
  ColorMesh genMesh(Eigen::VectorXf shapeCoeff,
                    Eigen::VectorXf expressionCoeff);

  ColorMesh genMesh(const double *shapeCoeff, int shapeCoeffSize,
                    const double *expressionCoeff, int expressionCoeffSize);

  /** Returns PCA basis matrix for shape */
  Eigen::MatrixXf getShapeBasisMatrix();

  /** Returns PCA basis matrix for expression */
  Eigen::MatrixXf getExpressionBasisMatrix();

  /** Returns deformation center for shape */
  Eigen::VectorXf getShapeDeformationCenter();

  /** Returns deformation center for expression */
  Eigen::VectorXf getExpressionDeformationCenter();

  /** Returns reference model vertex position vector */
  Eigen::VectorXf getReferenceVector();

  /** Returns the number of PCA shape model basis */
  int getShapeRank();

  /** Returns the number of PCA expression model basis */
  int getExpressionRank();

  /** Set/Get Landmark vertex index list */
  void setLandmarks(std::vector<int> lmk);
  std::vector<int> getLandmarks();
};
}; // namespace telef::face
