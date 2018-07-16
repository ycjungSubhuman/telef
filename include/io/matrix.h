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
    }
    template<class M>
    void readMat(const char *filename, M &mat) {
        std::ifstream f(filename, std::ios::binary);
        typename M::Index rows, cols;
        f.read((char*)(&rows), sizeof(typename M::Index));
        f.read((char*)(&cols), sizeof(typename M::Index));
        mat.resize(rows, cols);
        auto shapeRank = mat.cols();

        f.read((char*)mat.data(), rows*cols*sizeof(typename M::Scalar));
    }
}
