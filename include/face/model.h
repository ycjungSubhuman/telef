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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include "io/ply/meshio.h"
#include "type.h"
#include "util/eigen_pcl.h"
#define RANK 150

namespace fs = std::experimental::filesystem;

using namespace telef::mesh;

namespace {


    template <int Rank>
    Eigen::Matrix<float, Eigen::Dynamic, Rank> getPCABase(Eigen::MatrixXf data) {

        // Each row is a data. This is to match data matrix dimensions with formulas in Wikipedia.
        auto d = data.transpose();
        Eigen::MatrixXf centered = d.rowwise() - d.colwise().mean();
        // Fast singlular value computation using devide-and-conquer
        Eigen::BDCSVD<Eigen::MatrixXf> bdc(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Sort eigenvectors according to (singular value)^2 / (n -1), which is equal to eigenvalues
        std::vector<std::pair<float, Eigen::VectorXf>> pairs;
        if (d.rows() <= d.cols()) { //singular values are shorter than position dimension
            std::cout << "Singular values are shorter then dimension" << std::endl;
            pairs.resize(static_cast<unsigned long>(bdc.singularValues().rows()));
        }
        else { // singular values are exact match with V
            std::cout << "Exact match" << std::endl;
            pairs.resize(static_cast<unsigned long>(d.cols()));
        }
        std::cout << "Singluar Value **2" << std::endl;
        for(unsigned long i=0; i<pairs.size(); i++) {
            auto s = bdc.singularValues()(i);
            auto s2 = s*s;
            std::cout << s2 << ", ";
            pairs[i] = std::make_pair(
                    s2, // propertional to eigenvalue (omitted /(n-1))
                    bdc.matrixV().col(i)); // eivenvector, which is a PCA basis
        }
        std::cout << std::endl;
        std::sort(pairs.begin(), pairs.end(), [](auto &l, auto &r) {return l.first > r.first;});

        Eigen::Matrix<float, Eigen::Dynamic, Rank> result(d.cols(), Rank);
        for (int i = 0; i < std::min(Rank, static_cast<int>(bdc.singularValues().rows())); i++) {
            result.col(i).swap(pairs[i].second);
        }

        // Fill in empty basis if given PCA rank is higher than the rank of the data matrix
        if (Rank > d.cols()) {
            std::cout << "WARNING : given rank is higher than number of singular values" << std::endl;
            for (long i = d.cols(); i < Rank; i++) {
                result.col(i) = Eigen::VectorXf::Zero(d.cols());
            }
        }
        return result;
    }

    ColorMesh read(fs::path f) {
        if (f.extension().string() == ".ply") {
            return ColorMesh(telef::io::ply::readPlyMesh(f));
        }
        else {
            throw std::runtime_error("File " + f.string() + " is not supported");
        }
    }

    /** PCA model for deformations */
    template <int ShapeRank>
    class PCADeformationModel {
    public:
        Eigen::Matrix<float, Eigen::Dynamic, ShapeRank> shapeBase;
        Eigen::VectorXf mean;

        PCADeformationModel() = default;
        PCADeformationModel(Eigen::Matrix<float, Eigen::Dynamic, ShapeRank> shapeBase, Eigen::VectorXf mean)
        : shapeBase(shapeBase), mean(mean) {}


        explicit PCADeformationModel(std::vector<ColorMesh> &samples, ColorMesh &refMesh)
        {
            shapeBase.resize(samples.size(), ShapeRank);
            auto numData = samples.size();
            auto dataDim = refMesh.position.size();
            Eigen::MatrixXf positions(dataDim, numData);
            Eigen::MatrixXf colors(dataDim, numData);

            for(unsigned long i=0; i<samples.size(); i++) {
                auto mesh = samples[i];
                positions.col(i) = mesh.position.col(0) - refMesh.position.col(0);
            }


            shapeBase = getPCABase<ShapeRank>(positions);
            mean = positions.rowwise().mean();
        }

        Eigen::VectorXf genDeform(Eigen::Matrix<float, ShapeRank, 1> coeff) {
            if(coeff.rows() != shapeBase.cols()) {
                throw std::runtime_error("Coefficient dimension mismatch");
            }
            Eigen::VectorXf result = Eigen::VectorXf::Zero(shapeBase.rows());
            for (long i=0; i<ShapeRank; i++) {
                result += coeff(i) * shapeBase.col(i);
            }
            return mean + result;
        }

        Eigen::VectorXf genDeform(const double * const coeff, int size) {
            if(size != shapeBase.cols()) {
                throw std::runtime_error("Coefficient dimension mismatch");
            }
            Eigen::VectorXf result = Eigen::VectorXf::Zero(shapeBase.rows());
            for (long i=0; i<ShapeRank; i++) {
                result += coeff[i] * shapeBase.col(i);
            }
            return mean + result;
        }
    };

    template<class M>
    void writeMat(const char *filename, const M &mat) {
        std::ofstream f(filename, std::ios::binary);
        typename M::Index rows = mat.rows();
        typename M::Index cols = mat.cols();

        f.write((char*)(&rows), sizeof(typename M::Index));
        f.write((char*)(&cols), sizeof(typename M::Index));
        f.write((char*)mat.data(), rows*cols*sizeof(typename M::Scalar));
        f.close();
    }
    template<class M>
    void readMat(const char *filename, M &mat) {
        std::ifstream f(filename, std::ios::binary);
        typename M::Index rows, cols;
        f.read((char*)(&rows), sizeof(typename M::Index));
        f.read((char*)(&cols), sizeof(typename M::Index));
        if (mat.rows() != rows || mat.cols() != cols) {
            throw std::runtime_error("Load Fail (" + std::string(filename) + "): dimension mismatch");
        }
        f.read((char*)mat.data(), rows*cols*sizeof(typename M::Scalar));
        f.close();
    }

    void writeLmk(const char *filename, const std::vector<int> &lmk) {
        std::ofstream f(filename);
        f << lmk.size() << "\n\n";
        for (const auto &l : lmk) {
            f << l << "\n";
        }
        f.close();
    }

    void readLmk(const char *filename, std::vector<int> &lmk) {
        std::ifstream f(filename);
        std::string c;
        f >> c;
        int count = std::stoi(c);
        std::string pt;
        while(f >> pt) {
            lmk.push_back(std::stoi(pt));
        }
    }
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
            std::transform(f.begin(), f.end(), meshes.begin(), [](auto &a){return read(a);});

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

            Eigen::Matrix<float, Eigen::Dynamic, ShapeRank> shapeBase;
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

        Eigen::VectorXf getBasis(unsigned long coeffIndex) {
            return deformModel.shapeBase.col(coeffIndex);
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
