#pragma once

#include <string>
#include <vector>
#include <memory>
#include <experimental/filesystem>
#include <exception>
#include <algorithm>
#include <tuple>
#include <functional>
#include <random>
#include <ctime>

#include <Eigen/Dense>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include "type.h"
#include "mesh/mesh.h"
#include "face/deformation_model.h"
#include "io/ply/meshio.h"
#include "io/matrix.h"
#include "io/landmark.h"
#include "util/eigen_pcl.h"

#define RANK 40

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::mesh;
    using namespace telef::io;
}

namespace telef::face {

    /** PCA face model using PCA model of deformation btw reference mesh and samples*/
    template <int ShapeRank>
    class MorphableFaceModel {
    private:
        PCADeformationModel<ShapeRank> deformModel;
        ColorMesh refMesh;
        std::vector<int> landmarks;
        std::random_device rd;
        std::mt19937 mt;
    public:
        /** Construct PCA Model using a list of mesh files */
        MorphableFaceModel(std::vector<fs::path> &f, bool rigidAlignRequired=false)
        :mt(rd())
        {
            assert(f.size() > 0);
            std::vector<ColorMesh> meshes(f.size());
            std::transform(f.begin(), f.end(), meshes.begin(), [](auto &a){return telef::io::ply::readPlyMesh(a);});

            refMesh = meshes[0];
            if(rigidAlignRequired) {
                auto refCloud = util::convert(refMesh.position);
                auto reg = pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>();
                for (unsigned long i = 0; i < f.size(); i++) {
                    auto cloud = util::convert(meshes[i].position);
                    Eigen::Matrix4f trans;
                    reg.estimateRigidTransformation(*cloud, *refCloud, trans);
                    meshes[i].applyTransform(trans);
                }
            }
            deformModel = PCADeformationModel<ShapeRank>(meshes, refMesh);
            landmarks = std::vector<int>{1, 2, 3, 4, 5};
        };

        /** Load from existing model file */
        MorphableFaceModel(fs::path fileName): mt(rd()) {
            refMesh = telef::io::ply::readPlyMesh(fileName.string() + ".ref.ply");

            Eigen::MatrixXf shapeBase;
            Eigen::VectorXf mean;
            shapeBase.resize(refMesh.position.rows(), ShapeRank);
            mean.resize(refMesh.position.rows());
            readMat((fileName.string()+".deform.base").c_str(), shapeBase);
            readMat((fileName.string()+".deform.mean").c_str(), mean);
            deformModel = PCADeformationModel<ShapeRank>(shapeBase, mean);
            readLmk((fileName.string()+".lmk").c_str(), landmarks);
        }

        /** Save this model to a file */
        void save(fs::path fileName) {
            writeMat((fileName.string()+".deform.base").c_str(), deformModel.shapeBase);
            writeMat((fileName.string()+".deform.mean").c_str(), deformModel.mean);
            telef::io::ply::writePlyMesh(fileName.string() + ".ref.ply", refMesh);
            writeLmk((fileName.string()+".lmk").c_str(), landmarks);
        }

        /* Generate a xyzxyz... position vector using given coefficients */
        Eigen::VectorXf genPosition(Eigen::VectorXf shapeCoeff) {
            return refMesh.position + deformModel.genDeform(shapeCoeff);
        }

        Eigen::VectorXf genPosition(const double * const shapeCoeff, int size) {
            return refMesh.position + deformModel.genDeform(shapeCoeff, size);
        }

        /**
         * Generate a xyzxyz... position vector using given coefficients
         * Templated for Ceres
         */
        template <typename T>
        Eigen::Matrix<T, Eigen::Dynamic, 1> genPositionCeres(const T* const shapeCoeff, int size) {
            return refMesh.position.cast<T>() + deformModel.genDeformCeres(shapeCoeff, size);
        }

        ColorMesh genMesh(const double * const shapeCoeff, int size) {
            ColorMesh result;
            result.position = genPosition(shapeCoeff, size);
            result.triangles = refMesh.triangles;

            return result;
        }

        /* Generate a ColorMesh using given coefficients */
        ColorMesh genMesh(Eigen::VectorXf shapeCoeff) {
            ColorMesh result;
            result.position = genPosition(shapeCoeff);
            result.triangles = refMesh.triangles;

            return result;
        }

        Eigen::MatrixXf getBasisMatrix() {
            return deformModel.shapeBase;
        }

        Eigen::VectorXf getMeanDeformation() {
            return deformModel.mean;
        }

        Eigen::VectorXf getReferenceVector() {
            return refMesh.position;
        }

        int getRank() {
            return ShapeRank;
        }

        /* Generate a random sample ColorMesh */
        ColorMesh sample() {
            std::normal_distribution<float> dist(0.0, 0.005);

            std::vector<float> coeff(static_cast<unsigned long>(ShapeRank));
            std::generate(coeff.begin(), coeff.end(), [this, &dist]{return dist(this->mt);});
            float sum = 0.0f;
            for (auto &a : coeff) {
                sum += a;
            }

            assert(sum != 0.0f);
            for (int i=0; i<ShapeRank; i++) {
                coeff[i] /= sum;
                std::cout << coeff[i] << ", ";
            }
            std::cout << std::endl;

            return genMesh(Eigen::Map<Eigen::Matrix<float, ShapeRank, 1>>(coeff.data(), coeff.size()));
        }

        void setLandmarks(std::vector<int> lmk) {
            landmarks = lmk;
        }

        std::vector<int> getLandmarks() {
            return landmarks;
        }
    };
};
