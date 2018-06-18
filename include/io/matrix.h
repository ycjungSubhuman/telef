#pragma once

#include <fstream>
#include <exception>

#include <Eigen/Core>

namespace telef::io {
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
        if (mat.rows() != rows || mat.cols() > cols) {
            throw std::runtime_error("Load Fail (" + std::string(filename) + "): dimension mismatch");
        }
        bool resizeRequired = mat.cols() < cols; // If the shape rank of the matrix is smaller than the file
        auto shapeRank = mat.cols();

        if (resizeRequired) {
            // temporarily increase dimensions of the matrix
            mat.resize(rows, cols);
        }
        f.read((char*)mat.data(), rows*cols*sizeof(typename M::Scalar));

        if (resizeRequired) {
            Eigen::MatrixXf temp = mat.block(0, 0, rows, shapeRank);
            mat.resize(rows, shapeRank);
            mat = temp;
        }

        f.close();
    }
}
