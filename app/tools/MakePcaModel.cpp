#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>

#include "face/model.h"
#include "util/po_util.h"

namespace {
namespace po = boost::program_options;
using namespace telef::util;
} // namespace

int main(int argc, char **argv) {
  po::options_description desc("Constructs a PCA model from multiple meshes in "
                               "vertex-wise correspondence");
  desc.add_options()("help,H", "print this help message")(
      "ref,R", po::value<std::string>(),
      "specify a path of ply file for a PLY mesh model")(
      "shape,S", po::value<std::string>(),
      "specify a path of text file for a list of PLY mesh models with shape "
      "variance")("exp,E", po::value<std::string>(),
                  "specify a path of text file for a list of PLY mesh models "
                  "with expression variance")(
      "landmark,L", po::value<std::string>(),
      "specify a path of text file for a list of landmark indices")(
      "shaperank,P", po::value<int>(),
      "specify the maximum number of shape PCA basis(default 40)")(
      "exprank,X", po::value<int>(),
      "specify the maximum number of expression PCA basis(default 10)")(
      "output,O", po::value<std::string>(), "specify a output model path");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc << std::endl;
    return 1;
  }

  require(vm, "ref");
  require(vm, "shape");
  require(vm, "exp");
  require(vm, "landmark");
  require(vm, "output");

  int shapeRank = 40;
  int expressionRank = 10;

  if (vm.count("shaperank") > 0) {
    shapeRank = vm["shaperank"].as<int>();
  }
  if (vm.count("exprank") > 0) {
    expressionRank = vm["exprank"].as<int>();
  }

  std::ifstream shapeSampleStream(vm["shape"].as<std::string>());
  std::ifstream expressionSampleStream(vm["exp"].as<std::string>());
  std::string path;
  std::vector<fs::path> shapeSamplePaths;
  std::vector<fs::path> expressionSamplePaths;
  while (shapeSampleStream >> path) {
    std::cout << "Using Shape Sample: " << path << std::endl;
    shapeSamplePaths.emplace_back(path);
  }
  while (expressionSampleStream >> path) {
    std::cout << "Using Expresison Sample: " << path << std::endl;
    expressionSamplePaths.emplace_back(path);
  }

  telef::face::MorphableFaceModel model(
      fs::path(vm["ref"].as<std::string>()), shapeSamplePaths,
      expressionSamplePaths, fs::path(vm["landmark"].as<std::string>()),
      shapeRank, expressionRank);

  model.save(fs::path(vm["output"].as<std::string>()));
  return 0;
}