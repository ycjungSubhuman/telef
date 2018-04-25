#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace fs = std::experimental::filesystem;

int main() {
    telef::face::MorphableFaceModel<30> model(fs::path("data/lowrank"));
    auto arb = model.sample().triangles;
    Eigen::VectorXf coeff = Eigen::VectorXf::Zero(30);
    for(int i=1; i<=30; i++) {
        auto col = model.getBasis(i-1);
        ColorMesh mesh;
        mesh.position = col + model.genPosition(Eigen::VectorXf::Zero(30));
        mesh.triangles = arb;
        telef::io::ply::writePlyMesh(fs::path("samples/sample" + std::to_string(i) + ".ply"), mesh);
    }

    auto sample = model.genMesh(Eigen::VectorXf::Zero(30));
    telef::io::ply::writePlyMesh(fs::path("samples/sample_mean.ply"), sample);

    return 0;
}