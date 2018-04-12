#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"

namespace fs = std::experimental::filesystem;
int main() {
    auto path = fs::path("test.ply");

    auto mesh = telef::io::ply::readMesh(path);
    std::cout << mesh.position.size() << std::endl;
    std::cout << mesh.color.size() << std::endl;
    std::cout << mesh.triangles.size() << std::endl;
}