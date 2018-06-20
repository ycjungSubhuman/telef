#include <iostream>
#include <experimental/filesystem>

#include "face/model.h"

int main (int argc, char** argv) {
    std::vector<fs::path> files;

    if(argc < 3) {
        std::cout << "specify name, landmark" << std::endl;
    }

    std::string name(argv[1]);
    std::string lmk(argv[2]);

    for (int i=1; i<=10; i++) {
        files.push_back("/home/ycjung/Projects/kinect-face/data/"+ name + "/" + std::to_string(i) + ".ply");
    }
    telef::face::MorphableFaceModel<RANK> model(files, true);

    model.save(fs::path("/home/ycjung/Projects/kinect-face/data/"+name));

    //override landmark
    fs::copy(lmk, fs::path("/home/ycjung/Projects/kinect-face/data/"+name+".lmk"), fs::copy_options::overwrite_existing);

    return 0;
}