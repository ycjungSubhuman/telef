#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <iostream>
#include <ostream>
#include <string>

#include "align/rigid_pipe.h"
#include "cloud/cloud_pipe.h"
#include "feature/feature_detect_frontend.h"
#include "feature/feature_detect_pipe.h"
#include "io/device.h"
#include "io/frontend.h"
#include "io/grabber.h"
#include "io/merger/device_input_merger.h"
#include "io/ply/meshio.h"

#include "glog/logging.h"

#include "messages/messages.pb.h"

namespace {
using namespace telef::io;
using namespace telef::align;
using namespace telef::types;
using namespace telef::cloud;
using namespace telef::feature;

namespace fs = std::experimental::filesystem;

namespace po = boost::program_options;
} // namespace

/**
 * Project IntraFace landmark points onto captured pointcloud
 */

int main(int ac, const char *const *av) {
  po::options_description desc("Capture 3D Landmark Points from RGBD Camera "
                               "and Save into multiple CSV files");
  desc.add_options()("help,H", "print help message")(
      "detector,D",
      po::value<std::string>(),
      "specify Dlib pretrained Face detection model path")(
      "fake,F",
      po::value<std::string>(),
      "specify directory path to captured kinect frames")(
      "address,A",
      po::value<std::string>(),
      "specify server address for client to connect too");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  bool useFakeKinect = vm.count("fake") > 0;

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("detector") == 0) {
    std::cout << "Please specify 'detector path'" << std::endl;
    return 1;
  }

  if (vm.count("address") == 0) {
    std::cout << "Please specify 'server address' for client to connect too"
              << std::endl;
    return 1;
  }
  std::string detectModelPath;
  std::string fakePath("");
  std::string address("");

  detectModelPath = vm["detector"].as<std::string>();
  address = vm["address"].as<std::string>();

  if (useFakeKinect) {
    fakePath = vm["fake"].as<std::string>();
  }

  pcl::io::OpenNI2Grabber::Mode depth_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  pcl::io::OpenNI2Grabber::Mode image_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

  auto imagePipe = IdentityPipe<ImageT>();
  auto cloudPipe = FastBilateralFilterPipe(); // RemoveNaNPoints();

  auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>(
      [&imagePipe](auto in) -> decltype(auto) { return imagePipe(in); });
  auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>(
      [&cloudPipe](auto in) -> decltype(auto) { return cloudPipe(in); });

  auto viewFrontend = std::make_shared<Feature2DDetectFrontEnd>();
  auto faceDetector = DlibFaceDetectionPipe(detectModelPath);
  boost::asio::io_service ioService;
  auto featureDetector = FeatureDetectionClientPipe(address, ioService);

  auto pipe1 = compose(faceDetector, featureDetector);
  auto merger = std::make_shared<DeviceInputPipeMerger<FeatureDetectSuite>>(
      [&pipe1](auto in) -> decltype(auto) { return pipe1(in); });

  std::shared_ptr<ImagePointCloudDevice<
      DeviceCloudConstT,
      ImageT,
      DeviceInputSuite,
      FeatureDetectSuite>>
      device;

  if (useFakeKinect) {
    device = std::make_shared<FakeImagePointCloudDevice<
        DeviceCloudConstT,
        ImageT,
        DeviceInputSuite,
        FeatureDetectSuite>>(fs::path(fakePath), PlayMode::FPS_30);
  } else {
    auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);
    device = std::make_shared<ImagePointCloudDeviceImpl<
        DeviceCloudConstT,
        ImageT,
        DeviceInputSuite,
        FeatureDetectSuite>>(std::move(grabber), false);
  }

  device->setCloudChannel(cloudChannel);
  device->setImageChannel(imageChannel);
  device->addMerger(merger);
  merger->addFrontEnd(viewFrontend);

  device->run();

  featureDetector.disconnect();

  return 0;
}
