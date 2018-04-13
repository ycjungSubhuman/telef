#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace fs = std::experimental::filesystem;

int main() {
    std::vector<fs::path> files;
    for (int i=1; i<=50; i++) {
        files.push_back("/home/ycjung/Projects/flame-fitting/output/" + std::to_string(i) + ".ply");
    }
    telef::face::MorphableFaceModel<10> model(files);
    std::cout << "GENED" << std::endl;

    for(int i=1; i<10; i++) {
        auto sample = model.sample();
        telef::io::ply::writeMesh(fs::path("sample"+std::to_string(i)+".ply"), sample);
    }
}