#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace fs = std::experimental::filesystem;

int main() {
    telef::face::MorphableFaceModel<150> model(fs::path("data/example"));
    for(int i=1; i<10; i++) {
        auto sample = model.sample();
        telef::io::ply::writeMesh(fs::path("sample"+std::to_string(i)+".ply"), sample);
    }

    auto sample = model.genMesh(Eigen::VectorXf::Zero(150));
    telef::io::ply::writeMesh(fs::path("sample_mean.ply"), sample);

    return 0;
}