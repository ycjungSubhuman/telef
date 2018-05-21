#include "face/model.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::mesh;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> getPCABase(Eigen::MatrixXf data, int rank) {

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

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> result(d.cols(), rank);
        for (int i = 0; i < std::min(rank, static_cast<int>(bdc.singularValues().rows())); i++) {
            result.col(i).swap(pairs[i].second);
        }

        // Fill in empty basis if given PCA rank is higher than the rank of the data matrix
        if (rank > d.cols()) {
            std::cout << "WARNING : given rank is higher than number of singular values" << std::endl;
            for (long i = d.cols(); i < rank; i++) {
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
    /**
     * Read a matrix from file
     *
     * mat should be pre-allocated
     */
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
        std::string pt;
        while(f >> pt) {
            lmk.push_back(std::stoi(pt));
        }
    }
}

namespace telef::face {

    PCADeformationModel::PCADeformationModel(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> shapeBase,
                                             Eigen::VectorXf mean)
            : shapeBase(shapeBase), mean(mean) {}

    PCADeformationModel::PCADeformationModel(std::vector<ColorMesh> &samples, ColorMesh &refMesh, int shapeRank) :
            shapeRank(shapeRank)
    {
        auto numData = samples.size();
        auto dataDim = refMesh.position.size();
        Eigen::MatrixXf positions(dataDim, numData);
        Eigen::MatrixXf colors(dataDim, numData);

        for(unsigned long i=0; i<samples.size(); i++) {
            auto mesh = samples[i];
            positions.col(i) = mesh.position.col(0) - refMesh.position.col(0);
        }

        shapeBase = getPCABase(positions, shapeRank);
        mean = positions.rowwise().mean();
    }

    Eigen::VectorXf PCADeformationModel::genDeform(Eigen::Matrix<float, Eigen::Dynamic, 1> coeff) {
        if(coeff.rows() != shapeBase.cols()) {
            throw std::runtime_error("Coefficient dimension mismatch");
        }
        Eigen::VectorXf result = Eigen::VectorXf::Zero(shapeBase.rows());
        for (long i=0; i<shapeRank; i++) {
            result += coeff(i) * shapeBase.col(i);
        }
        return mean + result;
    }

    Eigen::VectorXf PCADeformationModel::genDeform(const double *const coeff, int size) {
        if(size != shapeBase.cols()) {
            throw std::runtime_error("Coefficient dimension mismatch");
        }
        Eigen::VectorXf result = Eigen::VectorXf::Zero(shapeBase.rows());
        for (long i=0; i<shapeRank; i++) {
            result += coeff[i] * shapeBase.col(i);
        }
        return mean + result;
    }

    MorphableFaceModel::MorphableFaceModel(std::vector<fs::path> &f, int shapeRank, bool rigidAlignRequired) :
            mt(rd()),
            shapeRank(shapeRank)
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
        deformModel = PCADeformationModel(meshes, refMesh, shapeRank);
        // TODO : make landmarks detected upon construction of Morphable Model
        landmarks = std::vector<int>{1}; // landmark placeholder
    }

    MorphableFaceModel::MorphableFaceModel(fs::path fileName) : mt(rd()) {
        refMesh = telef::io::ply::readPlyMesh(fileName.string() + ".ref.ply");

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> shapeBase;
        Eigen::VectorXf mean;
        shapeBase.resize(refMesh.position.rows(), shapeRank);
        mean.resize(refMesh.position.rows());
        readMat((fileName.string()+".deform.base").c_str(), shapeBase);
        readMat((fileName.string()+".deform.mean").c_str(), mean);
        deformModel = PCADeformationModel(shapeBase, mean);
        readLmk((fileName.string()+".lmk").c_str(), landmarks);
    }

    void MorphableFaceModel::save(fs::path fileName) {
        writeMat((fileName.string()+".deform.base").c_str(), deformModel.shapeBase);
        writeMat((fileName.string()+".deform.mean").c_str(), deformModel.mean);
        telef::io::ply::writePlyMesh(fileName.string() + ".ref.ply", refMesh);
        writeLmk((fileName.string()+".lmk").c_str(), landmarks);
    }

    Eigen::VectorXf MorphableFaceModel::genPosition(Eigen::VectorXf shapeCoeff) {
        return refMesh.position + deformModel.genDeform(shapeCoeff);
    }

    Eigen::VectorXf MorphableFaceModel::genPosition(const double *const shapeCoeff, int size) {
        return refMesh.position + deformModel.genDeform(shapeCoeff, size);
    }

    ColorMesh MorphableFaceModel::genMesh(const double *const shapeCoeff, int size) {
        ColorMesh result;
        result.position = genPosition(shapeCoeff, size);
        result.triangles = refMesh.triangles;

        return result;
    }

    ColorMesh MorphableFaceModel::genMesh(Eigen::VectorXf shapeCoeff) {
        ColorMesh result;
        result.position = genPosition(shapeCoeff);
        result.triangles = refMesh.triangles;

        return result;
    }

    Eigen::VectorXf MorphableFaceModel::getBasis(unsigned long coeffIndex) {
        return deformModel.shapeBase.col(coeffIndex);
    }

    int MorphableFaceModel::getRank() {
        return shapeRank;
    }

    ColorMesh MorphableFaceModel::sample() {
        std::normal_distribution<float> dist(0.0, 0.005);

        std::vector<float> coeff(static_cast<unsigned long>(shapeRank));
        std::generate(coeff.begin(), coeff.end(), [this, &dist]{return dist(this->mt);});
        float sum = 0.0f;
        for (auto &a : coeff) {
            sum += a;
        }

        assert(sum != 0.0f);
        for (int i=0; i<shapeRank; i++) {
            coeff[i] /= sum;
            std::cout << coeff[i] << ", ";
        }
        std::cout << std::endl;

        return genMesh(Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>(coeff.data(), coeff.size()));
    }

    void MorphableFaceModel::setLandmarks(std::vector<int> lmk) {
        landmarks = lmk;
    }

    std::vector<int> MorphableFaceModel::getLandmarks() {
        return landmarks;
    }
}
