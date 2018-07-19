#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <iostream>
#include <string>
#include <ostream>
#include <pcl/io/openni2_grabber.h>

#include "io/device.h"
#include "io/grabber.h"
#include "io/frontend.h"
#include "io/ply/meshio.h"
#include "io/merger/device_input_merger.h"
#include "feature/feature_detect_frontend.h"
#include "feature/feature_detect_pipe.h"
#include "cloud/cloud_pipe.h"
#include "align/rigid_pipe.h"

#include "glog/logging.h"

namespace {
    using namespace telef::io;
    using namespace telef::align;
    using namespace telef::types;
    using namespace telef::cloud;
    using namespace telef::feature;

    namespace fs = std::experimental::filesystem;

    namespace po = boost::program_options;
}

/**
 * Project IntraFace landmark points onto captured pointcloud
 */

int main(int ac, const char* const * av)
{
    po::options_description desc("Capture 3D Landmark Points from RGBD Camera and Save into multiple CSV files");
    desc.add_options()
            ("help,H", "print help message")
            ("detector,D", po::value<std::string>(), "specify Dlib pretrained Face detection model path")
            ("graph,G", po::value<std::string>(), "specify path to PRNet graph definition")
            ("checkpoint,C", po::value<std::string>(), "specify path to pretrained PRNet checkpoint")
            ("fake,F", po::value<std::string>(), "specify directory path to captured kinect frames");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    bool useFakeKinect = vm.count("fake") > 0;

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("detector") == 0) {
        std::cout << "Please specify 'detector path'"  << std::endl;
        return 1;
    }

    if (vm.count("graph") == 0) {
        std::cout << "graph specify 'PRNet Graph path'"  << std::endl;
        return 1;
    }

    if (vm.count("checkpoint") == 0) {
        std::cout << "Please specify 'PRNet Checkpoint path'"  << std::endl;
        return 1;
    }

    std::string detectModelPath;
    std::string prnetGraphPath;
    std::string prnetChkptPath;
    std::string fakePath("");

    detectModelPath = vm["detector"].as<std::string>();
    prnetGraphPath = vm["graph"].as<std::string>();
    prnetChkptPath = vm["checkpoint"].as<std::string>();

    if (useFakeKinect) {
        fakePath = vm["fake"].as<std::string>();
    }

    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

    auto imagePipe = IdentityPipe<ImageT>();
    auto cloudPipe = RemoveNaNPoints();

    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>([&imagePipe](auto in)->decltype(auto){return imagePipe(in);});
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>([&cloudPipe](auto in)->decltype(auto){return cloudPipe(in);});

    auto viewFrontend = std::make_shared<FeatureDetectFrontEnd>();
    auto faceDetector = DlibFaceDetectionPipe(detectModelPath);
    auto featureDetector = PRNetFeatureDetectionPipe(fs::path(prnetGraphPath), fs::path(prnetChkptPath));
    auto lmkToScanFitting = LmkToScanRigidFittingPipe();


    auto pipe1 = compose(faceDetector, featureDetector, lmkToScanFitting);
    auto merger = std::make_shared<DeviceInputPipeMerger<FittingSuite >>([&pipe1](auto in)->decltype(auto){return pipe1(in);});

    std::shared_ptr<ImagePointCloudDevice<DeviceCloudConstT, ImageT, DeviceInputSuite, FittingSuite>> device = NULL;

    if (useFakeKinect) {
        device = std::make_shared<FakeImagePointCloudDevice <
                DeviceCloudConstT,
                ImageT,
                DeviceInputSuite,
                FittingSuite>>(fs::path(fakePath), PlayMode::FPS_30);
    } else {
        auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);
        device = std::make_shared<ImagePointCloudDeviceImpl<
                DeviceCloudConstT,
                ImageT,
                DeviceInputSuite,
                FittingSuite>>(std::move(grabber), false);
    }

    device->setCloudChannel(cloudChannel);
    device->setImageChannel(imageChannel);
    device->addMerger(merger);
    merger->addFrontEnd(viewFrontend);

    device->run();

    return 0;
}

