#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace fs = std::experimental::filesystem;

int main() {
    std::vector<fs::path> files;
    for (int i=1; i<=4000; i++) {
        files.push_back("/home/ycjung/Projects/flame-fitting/output/" + std::to_string(i) + ".ply");
    }
    telef::face::MorphableFaceModel<150> model(files);

    for(int i=1; i<10; i++) {
        auto sample = model.sample();
        telef::io::ply::writeMesh(fs::path("sample"+std::to_string(i)+".ply"), sample);
    }
    model.save(fs::path("data/example"));

    telef::face::MorphableFaceModel<150> model2(fs::path("data/example"));

    for(int i=1; i<10; i++) {
        auto sample = model2.sample();
        telef::io::ply::writeMesh(fs::path("newsample"+std::to_string(i)+".ply"), sample);
    }

    telef::face::MorphableClusterModel<150> model3(files);

    return 0;
}