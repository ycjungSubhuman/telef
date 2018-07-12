#include "face/model.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::mesh;
    using namespace telef::io;
}

namespace telef::face {
    MorphableFaceModel::MorphableFaceModel(fs::path refSamplePath, const std::vector<fs::path> &shapeSamplePaths,
                                           const std::vector<fs::path> &expressionSamplePaths, fs::path landmarkIdxPath,
                                           int shapeRank, int expressionRank)
            : mt(rd()), shapeRank(shapeRank), expressionRank(expressionRank)
    {
        assert(shapeSamplePaths.size() > 0);
        assert(expressionSamplePaths.size() > 0);
        refMesh = telef::io::ply::readPlyMesh(refSamplePath);
        std::vector<ColorMesh> shapeSamples(shapeSamplePaths.size());
        std::vector<ColorMesh> expressionSamples(expressionSamplePaths.size());
        std::transform(shapeSamplePaths.begin(), shapeSamplePaths.end(), shapeSamples.begin(),
                       [](auto &a){return telef::io::ply::readPlyMesh(a);});
        std::transform(expressionSamplePaths.begin(), expressionSamplePaths.end(), expressionSamples.begin(),
                       [](auto &a){return telef::io::ply::readPlyMesh(a);});

        shapeModel = PCADeformationModel(shapeSamples, refMesh, shapeRank);
        expressionModel = PCADeformationModel(expressionSamples, refMesh, expressionRank);
        readLmk(landmarkIdxPath.c_str(), landmarks);
    }

    MorphableFaceModel::MorphableFaceModel(fs::path fileName) : mt(rd()) {
        refMesh = telef::io::ply::readPlyMesh(fileName.string() + ".ref.ply");

        Eigen::MatrixXf shapeBase, expressionBase;
        Eigen::VectorXf shapeMean, expressionMean;
        readMat((fileName.string()+".shape.base").c_str(), shapeBase);
        readMat((fileName.string()+".shape.mean").c_str(), shapeMean);
        readMat((fileName.string() + ".exp.base").c_str(), expressionBase);
        readMat((fileName.string() + ".exp.mean").c_str(), expressionMean);
        shapeRank = static_cast<int>(shapeBase.cols());
        expressionRank = static_cast<int>(expressionBase.cols());
        shapeModel = PCADeformationModel(shapeBase, shapeMean, shapeRank);
        expressionModel = PCADeformationModel(expressionBase, expressionMean, expressionRank);
        readLmk((fileName.string()+".lmk").c_str(), landmarks);
    }

    void MorphableFaceModel::save(fs::path fileName) {
        writeMat((fileName.string()+".shape.base").c_str(), shapeModel.pcaBasisVectors);
        writeMat((fileName.string()+".exp.base").c_str(), expressionModel.pcaBasisVectors);
        writeMat((fileName.string()+".shape.mean").c_str(), shapeModel.mean);
        writeMat((fileName.string()+".exp.mean").c_str(), expressionModel.mean);
        telef::io::ply::writePlyMesh(fileName.string() + ".ref.ply", refMesh);
        writeLmk((fileName.string()+".lmk").c_str(), landmarks);
    }

    Eigen::VectorXf MorphableFaceModel::genPosition(Eigen::VectorXf shapeCoeff, Eigen::VectorXf expressionCoeff) {
        return refMesh.position
               + shapeModel.genDeform(shapeCoeff)
               + expressionModel.genDeform(expressionCoeff);
    }

    Eigen::VectorXf
    MorphableFaceModel::genPosition(const double *shapeCoeff, int shapeCoeffSize, const double *expressionCoeff,
                                    int expressionCoeffSize) {
        return refMesh.position
               + shapeModel.genDeform(shapeCoeff, shapeCoeffSize)
               + expressionModel.genDeform(expressionCoeff, expressionCoeffSize);
    }

    ColorMesh MorphableFaceModel::genMesh(Eigen::VectorXf shapeCoeff, Eigen::VectorXf expressionCoeff) {
        ColorMesh result;
        result.position = genPosition(shapeCoeff, expressionCoeff);
        result.triangles = refMesh.triangles;

        return result;
    }

    ColorMesh MorphableFaceModel::genMesh(const double *shapeCoeff, int shapeCoeffSize, const double *expressionCoeff,
                                          int expressionCoeffSize) {
        ColorMesh result;
        result.position = genPosition(shapeCoeff, shapeCoeffSize, expressionCoeff, expressionCoeffSize);
        result.triangles = refMesh.triangles;

        return result;
    }

    Eigen::MatrixXf MorphableFaceModel::getShapeBasisMatrix() {
        return shapeModel.pcaBasisVectors;
    }

    Eigen::MatrixXf MorphableFaceModel::getExpressionBasisMatrix() {
        return expressionModel.pcaBasisVectors;
    }

    Eigen::VectorXf MorphableFaceModel::getMeanShapeDeformation() {
        return shapeModel.mean;
    }

    Eigen::VectorXf MorphableFaceModel::getMeanExpressionDeformation() {
        return expressionModel.mean;
    }

    Eigen::VectorXf MorphableFaceModel::getReferenceVector() {
        return refMesh.position;
    }

    int MorphableFaceModel::getShapeRank() {
        return shapeRank;
    }

    int MorphableFaceModel::getExpressionRank() {
        return expressionRank;
    }

    void MorphableFaceModel::setLandmarks(std::vector<int> lmk) {
        landmarks = lmk;
    }

    std::vector<int> MorphableFaceModel::getLandmarks() {
        return landmarks;
    }
}

