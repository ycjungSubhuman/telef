#pragma once

#include <experimental/filesystem>
#include <vector>
#include <exception>

#include <Eigen/Core>

#include "mesh/mesh.h"
#include "io/ply/YaPly.h"

using namespace telef::mesh;
namespace fs = std::experimental::filesystem;

namespace telef::io::ply {
    ColorMesh readMesh(fs::path f);
    void writeMesh(fs::path f, ColorMesh &mesh);
}

