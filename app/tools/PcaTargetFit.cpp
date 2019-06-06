#include <boost/program_options.hpp>
#include <cuda_runtime_api.h>
#include <experimental/filesystem>
#include <feature/feature_detect_pipe.h>
#include <ostream>
#include <string>

#include "align/nonrigid_pipe.h"
#include "align/rigid_pipe.h"
#include "align/lmkfit_pipe.h"
#include "align/bsfit_pipe.h"
#include "cloud/cloud_pipe.h"
#include "face/feeder.h"
#include "feature/feature_detector.h"
#include "io/align/align_frontend.h"
#include "io/device.h"
#include "io/frontend.h"
#include "io/grabber.h"
#include "io/merger.h"
#include "io/merger/device_input_merger.h"
#include "io/ply/meshio.h"
#include "io/normaldepth_pipe.h"
#include "io/wrapper.h"
#include "vis/fitting_visualizer.h"
#include "intrinsic/intrinsic_pipe.h"

#include "glog/logging.h"
#include "mesh/color_projection_pipe.h"
#include "mesh/mesh.h"
#include "util/cudautil.h"
#include "util/po_util.h"

namespace {
using namespace telef::io::align;
using namespace telef::io;
using namespace telef::cloud;
using namespace telef::align;
using namespace telef::face;
using namespace telef::mesh;
using namespace telef::vis;
using namespace telef::util;
using namespace telef::intrinsic;

namespace fs = std::experimental::filesystem;

namespace po = boost::program_options;
} // namespace

/**
 *   -name1
 *   path1
 *   path2
 *   ...
 *
 *   -name2
 *   path1
 *   path2
 *   ...
 *
 */
std::vector<std::pair<std::string, fs::path>> readGroups(fs::path p) {
  std::ifstream file(p);

  std::vector<std::pair<std::string, fs::path>> result;

  while (!file.eof()) {
    std::string word;
    file >> word;
    if (*word.begin() == '-') // name of group
    {
      std::string p;
      file >> p;
      result.push_back(std::make_pair(word, p));
    }
  }

  file.close();
  return result;
}

/** Loads an RGB image and a corresponding pointcloud. Make and write PLY face
 * mesh out of it. */
int main(int ac, const char *const *av) {
  google::InitGoogleLogging(av[0]);

  float *d;
  CUDA_CHECK(cudaMalloc((void **)(&d), sizeof(float)));

  po::options_description desc("Captures RGB-D from camera. Generate and write "
                               "face mesh as ply and obj");
  desc.add_options()("help,H", "print this help message")(
      "models,M", po::value<std::vector<std::string>>()->multitoken(), "specify PCA model path")(
      "detector,D",
      po::value<std::string>(),
      "specify Dlib pretrained Face detection model path")(
      "vis,V", "run visualizer")(
      "depthnormal,T",
      po::value<std::string>(),
      "record depth normal image to specified directory")(
      "skip-intrinsic", "Does not perform intrinsic decomposition")(
      "reg", po::value<float>()->default_value(0.00001f), "Regularizer")(
      "geo,Z", "Adds Geometric Term")(
      "geo-weight,W", po::value<float>(), "Weight control for Geometric Term")(
      "geo-radius,R",
      po::value<float>(),
      "Search Radius for Mesh to Scan correspondance")(
      "geo-max-points,P",
      po::value<int>(),
      "Max Number of points used in Geometric Term")(
      "fake,F",
      po::value<std::string>(),
      "specify directory path to captured kinect frames")(
      "fake-loop", "If specified, loops in fake mode")(
      "bilaterFilter,B", "Use BilaterFilter on depth scan")(
      "bi-sigmaS,S", po::value<float>(), "BilaterFilter spatial width")(
      "bi-sigmaR,Q", po::value<float>(), "BilaterFilter range sigma")(
      "UsePrevFrame,U",
      "Use previous frames fitted parameters to increase performance")(
      "address,A",
      po::value<std::string>(),
      "specify server address for client to connect too");
  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc << std::endl;
    return 1;
  }

  require(vm, "models");
  require(vm, "detector");

  std::vector<std::string> modelPaths = vm["models"].as<std::vector<std::string>>();
  std::string detectModelPath = vm["detector"].as<std::string>();
  std::string address("");

  if (vm.count("address") == 0) {
    std::cout << "Please specify 'server address' for client to connect too"
              << std::endl;
    return 1;
  }

  address = vm["address"].as<std::string>();

  bool usePrevFrame = vm.count("UsePrevFrame") > 0;
  float geoWeight, geoSearchRadius;
  int geoMaxPoints;
  bool addGeoTerm = vm.count("geo") > 0;
  if (addGeoTerm) {
    geoWeight = vm["geo-weight"].as<float>();
    geoSearchRadius = vm["geo-radius"].as<float>();
    geoMaxPoints = vm["geo-max-points"].as<int>();
    std::cout << "Adding Geo Term..." << std::endl;
  }

  float biSigmaS = 5;
  float biSigmaR = 5e-3;
  bool useBilaterlFilter = vm.count("bilaterFilter") > 0;
  if (useBilaterlFilter) {
    std::cout << "Adding BilaterFilter... " << std::endl;
    if (vm.count("bi-sigmaS") > 0) {
      biSigmaS = vm["bi-sigmaS"].as<float>();
    }
    if (vm.count("bi-sigmaR") > 0) {
      biSigmaR = vm["bi-sigmaR"].as<float>();
    }
  }

  std::string fakePath("");
  bool useFakeKinect = vm.count("fake") > 0;
  if (useFakeKinect) {
    fakePath = vm["fake"].as<std::string>();
  }

  pcl::io::OpenNI2Grabber::Mode depth_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  pcl::io::OpenNI2Grabber::Mode image_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  auto imagePipe = IdentityPipe<ImageT>();

  Pipe<DeviceCloudConstT, DeviceCloudConstT> *cloudPipe;
  if (useBilaterlFilter) {
    cloudPipe = new FastBilateralFilterPipe(biSigmaS, biSigmaR);
  } else {
    cloudPipe = new IdentityPipe<DeviceCloudConstT>();
  }
  auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>(
      [&imagePipe](auto in) -> decltype(auto) { return imagePipe(in); });
  auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>(
      [cloudPipe](auto in) -> decltype(auto) { return (*cloudPipe)(in); });

  auto rigid = PCARigidFittingPipe();
  auto lmkfit = LmkFitPipe(vm["reg"].as<float>());
  auto bsfit = BsFitPipe();
  /*
  auto nonrigid = PCAGPUNonRigidFittingPipe(
      geoWeight, geoMaxPoints, geoSearchRadius, addGeoTerm, usePrevFrame);
  */
  auto toresult = PCAToFittingResultPipe();
  auto fitting2Projection = Fitting2ProjectionPipe();
  auto colorProjection = ColorProjectionPipe();

  std::vector<std::shared_ptr<MorphableFaceModel>> models;
  for(size_t i=0; i<modelPaths.size(); i++)
  {
    models.emplace_back(new MorphableFaceModel(fs::path(modelPaths[i].c_str())));
  }
  auto normaldepth = MeshNormalDepthRendererWrapper();
  auto modelFeeder = MultipleModelFeederPipe(models, normaldepth, vm["reg"].as<float>());
  std::shared_ptr<DeviceInputPipeMerger<PCANonRigidFittingResult>> merger;
  auto faceDetector = DlibFaceDetectionPipe(detectModelPath);

  boost::asio::io_service ioService;
  auto featureDetector = FeatureDetectionClientPipe(address, ioService);
  auto intrinsic = IntrinsicPipe();

  auto lmkToScanFitting = LmkToScanRigidFittingPipe();
  auto pipe1 = compose(
      faceDetector,
      featureDetector,
      lmkToScanFitting,
      modelFeeder,
      rigid,
      lmkfit,
      rigid,
      bsfit,
      toresult,
      normaldepth);

  bool is_intrinsic = 0 == vm.count("skip-intrinsic");
  merger = std::make_shared<DeviceInputPipeMerger<PCANonRigidFittingResult>>(
      [&pipe1, is_intrinsic, &intrinsic](auto in) -> decltype(auto) {
        auto res = pipe1(in);
        if(is_intrinsic) return intrinsic(res);
        else return res;
      });
  if (vm.count("vis") > 0) {
    auto frontend =
        std::make_shared<FittingVisualizer>(geoMaxPoints, geoSearchRadius);
    merger->addFrontEnd(frontend);
  }

  if (vm.count("depthnormal") > 0) {
    auto frontend = std::make_shared<DepthNormalFrontend>(
        vm["depthnormal"].as<std::string>());
    merger->addFrontEnd(frontend);
  }

  std::shared_ptr<ImagePointCloudDevice<
      DeviceCloudConstT,
      ImageT,
      DeviceInputSuite,
      PCANonRigidFittingResult>>
      device = NULL;

  if (useFakeKinect) {
    const auto playmode =
        (0 < vm.count("fake-loop")) ? PlayMode::FPS_30_LOOP : PlayMode::FPS_30;
    device = std::make_shared<FakeImagePointCloudDevice<
        DeviceCloudConstT,
        ImageT,
        DeviceInputSuite,
        PCANonRigidFittingResult>>(fs::path(fakePath), playmode);
  } else {
    auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);
    device = std::make_shared<ImagePointCloudDeviceImpl<
        DeviceCloudConstT,
        ImageT,
        DeviceInputSuite,
        PCANonRigidFittingResult>>(std::move(grabber), false);
  }

  device->setCloudChannel(cloudChannel);
  device->setImageChannel(imageChannel);
  device->addMerger(merger);
  device->run();

  free(cloudPipe);
  cloudPipe = NULL;
  featureDetector.disconnect();

  return 0;
}
