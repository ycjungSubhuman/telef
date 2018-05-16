#pragma once

#include <experimental/filesystem>
#include <vector>
#include <exception>

#include <Eigen/Core>

#include "mesh/mesh.h"
#include "io/ply/YaPly.h"

namespace {
    using namespace telef::mesh;
    namespace fs = std::experimental::filesystem;
}

namespace telef::io::ply {
    ColorMesh readPlyMesh(fs::path f);
    void writePlyMesh(fs::path f, ColorMesh &mesh);
    void writeObjMesh(fs::path f, ColorMesh &mesh);
}

