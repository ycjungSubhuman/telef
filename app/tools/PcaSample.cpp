#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace {
    namespace fs = std::experimental::filesystem;
}

int main() {
    telef::face::MorphableFaceModel<SHAPE_RANK> model(fs::path("data/pcamodel"));
    Eigen::VectorXf coeff = Eigen::VectorXf::Zero(SHAPE_RANK);
    for(int i=1; i<=10; i++) {
        auto m = model.sample();
        telef::io::ply::writePlyMesh(fs::path("samples/")/fs::path(std::to_string(i)+".ply"), m);
    }

    auto sample = model.genMesh(Eigen::VectorXf::Zero(SHAPE_RANK));
    telef::io::ply::writePlyMesh(fs::path("samples/sample_mean.ply"), sample);

    return 0;
}