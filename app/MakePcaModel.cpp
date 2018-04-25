#include <iostream>
#include <experimental/filesystem>

#include "face/model.h"

int main (int argc, char** argv) {
    std::vector<fs::path> files;
    for (int i=1; i<=10; i++) {
        files.push_back("/home/ycjung/Downloads/Export/" + std::to_string(i) + ".ply");
    }
    telef::face::MorphableFaceModel<5> model(files, true);

    model.save(fs::path("/home/ycjung/Downloads/Export/pcamodel"));

    return 0;
}