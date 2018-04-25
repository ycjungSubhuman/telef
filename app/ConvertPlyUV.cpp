#include <iostream>
#include <experimental/filesystem>

#include "io/ply/meshio.h"


int main (int argc, char** argv) {

    if(argc <= 1) {
        std::cout << "Provide file name" << std::endl;
        return 1;
    }

    auto f = std::experimental::filesystem::path(argv[1]);
    auto mesh = telef::io::ply::readPlyMesh(f);
    telef::io::ply::writeObjMesh((f.parent_path()/f.stem()).string()+".obj", mesh);

    return 0;
}