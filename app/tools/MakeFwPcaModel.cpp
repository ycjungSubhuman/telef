#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>

#include "face/model.h"
#include "io/dataset/face_warehouse.h"
#include "io/landmark.h"
#include "io/ply/meshio.h"
#include "util/po_util.h"

namespace {
namespace po = boost::program_options;
using namespace telef::util;
using namespace telef::io;
using namespace telef::face;
using namespace telef::io::dataset;
namespace fs = std::experimental::filesystem;
} // namespace

int main(int argc, char **argv) {
  po::options_description desc("Constructs a PCA model from FaceWarehouse");
  desc.add_options()("help,H", "print this help message")(
      "fw,F", po::value<std::string>(), "FaceWarehouse root directory path")(
      "rank,R",
      po::value<int>()->default_value(50),
      "Set maximum PCA basis for shape")(
      "landmark,L", po::value<std::string>(), "Landmark vertex index file")(
      "output,O", po::value<std::string>(), "specify a output model path");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc << std::endl;
    return 1;
  }

  require(vm, "fw");
  require(vm, "landmark");
  require(vm, "rank");

  std::vector<int> lmkInds;
  std::string lmkPath = vm["landmark"].as<std::string>();
  readLmk(lmkPath.c_str(), lmkInds);

  FaceWarehouseAllSampler sampler;
  FaceWarehouse fw(vm["fw"].as<std::string>(), lmkInds);

  const auto ref = fw.GetRef();
  const auto idSamples = sampler.SampleId(fw);
  const auto exSamples = sampler.SampleEx(fw);

  const auto shapeModel = std::make_shared<PCADeformationModel>(
      idSamples, ref, vm["rank"].as<int>());
  const auto expModel = std::make_shared<BlendShapeDeformationModel>(
      exSamples, ref, exSamples.size());

  MorphableFaceModel model(ref, shapeModel, expModel, lmkInds);
  model.save(fs::path(vm["output"].as<std::string>()));

  MorphableFaceModel m2(vm["output"].as<std::string>());
  for(size_t i=0; i<exSamples.size(); i++)
    {
      Eigen::VectorXf sh = Eigen::VectorXf::Zero(idSamples.size());
      Eigen::VectorXf ex = Eigen::VectorXf::Zero(exSamples.size());
      ex(i) = 1.0f;
      auto bs = m2.genMesh(sh, ex);
      telef::io::ply::writePlyMesh("__"+std::to_string(i)+".ply", bs);
    }

  return 0;
}
