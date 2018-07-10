#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <boost/program_options.hpp>

#include "face/model.h"

namespace po = boost::program_options;

void require(const po::variables_map &vm, const std::string argname) {
    if(vm.count(argname) == 0) {
        std::cout << "Please specify " << argname << std::endl;
        std::cout << "Give '--help' option to see all available options" << argname << std::endl;
        exit(1);
    }
}

int main (int argc, char** argv) {
    po::options_description desc("Constructs a PCA model from multiple meshes in vertex-wise correspondence");
    desc.add_options()
            ("help,H", "print this help message")
            ("filelist,F", po::value<std::string>(), "specify a path of text file for a list of PLY mesh models")
            ("landmark,L", po::value<std::string>(), "specify a path of text file for a list of landmark indices")
            ("output,O", po::value<std::string>(), "specify a output model path");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help") > 0) {
        std::cout << desc << std::endl;
        return 1;
    }

    require(vm, "filelist");
    require(vm, "landmark");
    require(vm, "output");

    std::ifstream flist_stream(vm["filelist"].as<std::string>());

    std::string path;
    std::vector<fs::path> files;
    while(flist_stream >> path) {
        std::cout << "Using : " << path << std::endl;
        files.emplace_back(path);
    }

    telef::face::MorphableFaceModel<RANK> model(files, true);

    model.save(fs::path(vm["output"].as<std::string>()));

    //override landmark
    fs::copy(fs::path(vm["landmark"].as<std::string>()),
            fs::path(vm["output"].as<std::string>() + ".lmk"), fs::copy_options::overwrite_existing);

    return 0;
}