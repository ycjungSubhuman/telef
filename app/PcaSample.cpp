#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace fs = std::experimental::filesystem;

int main() {
    telef::face::MorphableFaceModel<150> model(fs::path("data/example"));
    Eigen::VectorXf coeff = Eigen::VectorXf::Zero(150);
    for(int i=1; i<100; i++) {
        auto sample = model.genMesh(coeff);
        telef::io::ply::writePlyMesh(fs::path("samples/sample" + std::to_string(i) + ".ply"), sample);
        coeff[0] += 0.001;
    }

    auto sample = model.genMesh(Eigen::VectorXf::Zero(150));
    telef::io::ply::writePlyMesh(fs::path("samples/sample_mean.ply"), sample);

    return 0;
}