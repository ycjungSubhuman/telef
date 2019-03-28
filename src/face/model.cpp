#include "face/model.h"

namespace {
namespace fs = std::experimental::filesystem;
using namespace telef::mesh;
using namespace telef::io;
} // namespace

namespace telef::face {
MorphableFaceModel::MorphableFaceModel(
    fs::path refSamplePath,
    const std::vector<fs::path> &shapeSamplePaths,
    const std::vector<fs::path> &expressionSamplePaths,
    fs::path landmarkIdxPath,
    int shapeRank,
    int expressionRank)
    : mt(rd()) {
  assert(shapeSamplePaths.size() > 0);
  assert(expressionSamplePaths.size() > 0);
  refMesh = telef::io::ply::readPlyMesh(refSamplePath);
  std::vector<ColorMesh> shapeSamples(shapeSamplePaths.size());
  std::vector<ColorMesh> expressionSamples(expressionSamplePaths.size());
  std::transform(
      shapeSamplePaths.begin(),
      shapeSamplePaths.end(),
      shapeSamples.begin(),
      [](auto &a) { return telef::io::ply::readPlyMesh(a); });
  std::transform(
      expressionSamplePaths.begin(),
      expressionSamplePaths.end(),
      expressionSamples.begin(),
      [](auto &a) { return telef::io::ply::readPlyMesh(a); });

  shapeModel =
      std::make_shared<PCADeformationModel>(shapeSamples, refMesh, shapeRank);
  expressionModel = std::make_shared<BlendShapeDeformationModel>(
      expressionSamples, refMesh, expressionRank);
  readLmk(landmarkIdxPath.c_str(), landmarks);
}

MorphableFaceModel::MorphableFaceModel(fs::path fileName) : mt(rd()) {
  refMesh = telef::io::ply::readPlyMesh(fileName.string() + ".ref.ply");
  shapeModel = std::make_shared<PCADeformationModel>(
      fs::path(fileName.string() + ".shape"));
  expressionModel = std::make_shared<BlendShapeDeformationModel>(
      fs::path(fileName.string() + ".exp"));
  readLmk((fileName.string() + ".lmk").c_str(), landmarks);
}

void MorphableFaceModel::save(fs::path fileName) {
  shapeModel->save(fileName.string() + ".shape");
  expressionModel->save(fileName.string() + ".exp");
  telef::io::ply::writePlyMesh(fileName.string() + ".ref.ply", refMesh);
  writeLmk((fileName.string() + ".lmk").c_str(), landmarks);
}

Eigen::VectorXf MorphableFaceModel::genPosition(
    Eigen::VectorXf shapeCoeff, Eigen::VectorXf expressionCoeff) {
  return refMesh.position + shapeModel->genDeform(shapeCoeff) +
      expressionModel->genDeform(expressionCoeff);
}

Eigen::VectorXf MorphableFaceModel::genPosition(
    const double *shapeCoeff,
    int shapeCoeffSize,
    const double *expressionCoeff,
    int expressionCoeffSize) {
  return refMesh.position + shapeModel->genDeform(shapeCoeff, shapeCoeffSize) +
      expressionModel->genDeform(expressionCoeff, expressionCoeffSize);
}

ColorMesh MorphableFaceModel::genMesh(
    Eigen::VectorXf shapeCoeff, Eigen::VectorXf expressionCoeff) {
  ColorMesh result;
  result.position = genPosition(shapeCoeff, expressionCoeff);
  result.triangles = refMesh.triangles;

  return result;
}

ColorMesh MorphableFaceModel::genMesh(
    const double *shapeCoeff,
    int shapeCoeffSize,
    const double *expressionCoeff,
    int expressionCoeffSize) {
  ColorMesh result;
  result.position = genPosition(
      shapeCoeff, shapeCoeffSize, expressionCoeff, expressionCoeffSize);
  result.triangles = refMesh.triangles;

  return result;
}

Eigen::MatrixXf MorphableFaceModel::getShapeBasisMatrix() {
  return shapeModel->getBasisMatrix();
}

Eigen::MatrixXf MorphableFaceModel::getExpressionBasisMatrix() {
  return expressionModel->getBasisMatrix();
}

Eigen::VectorXf MorphableFaceModel::getShapeDeformationCenter() {
  return shapeModel->getCenter();
}

Eigen::VectorXf MorphableFaceModel::getExpressionDeformationCenter() {
  return expressionModel->getCenter();
}

Eigen::VectorXf MorphableFaceModel::getReferenceVector() {
  return refMesh.position;
}

int MorphableFaceModel::getShapeRank() { return shapeModel->getRank(); }

int MorphableFaceModel::getExpressionRank() {
  return expressionModel->getRank();
}

void MorphableFaceModel::setLandmarks(std::vector<int> lmk) { landmarks = lmk; }

std::vector<int> MorphableFaceModel::getLandmarks() { return landmarks; }
} // namespace telef::face
