#include <iostream>
#include <experimental/filesystem>
#include <boost/program_options.hpp>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace fs = std::experimental::filesystem;
namespace po = boost::program_options;

int main() {
    auto path = fs::path("test.ply");

    auto mesh = telef::io::ply::readMesh(path);
    std::cout << mesh.position.size() << std::endl;
    std::cout << mesh.color.size() << std::endl;
    std::cout << mesh.triangles.size() << std::endl;

    telef::io::ply::writeMesh(fs::path("testout.ply"), mesh);
}