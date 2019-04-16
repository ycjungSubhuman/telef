#pragma once

#include <exception>
#include <experimental/filesystem>
#include <vector>

#include <Eigen/Core>

#include "io/ply/YaPly.h"
#include "mesh/mesh.h"

namespace {
using namespace telef::mesh;
namespace fs = std::experimental::filesystem;
} // namespace

namespace telef::io::ply {
ColorMesh readPlyMesh(fs::path f);
void writePlyMesh(fs::path f, ColorMesh &mesh);
void writeObjMesh(fs::path f, fs::path img_path, ColorMesh &mesh);
} // namespace telef::io::ply
