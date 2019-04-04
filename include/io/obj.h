#pragma once

#include <Eigen/Core>
#include <experimental/filesystem>

namespace {
namespace fs = std::experimental::filesystem;
}

namespace telef::io {

/**
 * Read obj file and returns eigen matrices
 */
std::pair<Eigen::MatrixXf, Eigen::MatrixXi> readObj(const fs::path &p);
} // namespace telef::io
