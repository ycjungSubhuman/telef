#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <string>
#include <ostream>
#include <cuda_runtime_api.h>

#include "align/rigid_pipe.h"
#include "align/nonrigid_pipe.h"
#include "face/classify_pipe.h"
#include "feature/feature_detect_pipe.h"
#include "io/device.h"
#include "io/grabber.h"
#include "io/merger/device_input_merger.h"
#include "io/frontend.h"
#include "io/ply/meshio.h"
#include "io/align/align_frontend.h"
#include "cloud/cloud_pipe.h"

#include "mesh/mesh.h"
#include "mesh/color_projection_pipe.h"
#include "glog/logging.h"
#include "util/cudautil.h"

namespace {
    using namespace telef::io::align;
    using namespace telef::io;
    using namespace telef::cloud;
    using namespace telef::align;
    using namespace telef::face;
    using namespace telef::mesh;

    namespace fs = std::experimental::filesystem;

    namespace po = boost::program_options;
}

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

    while(!file.eof()) {
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

/** Loads an RGB image and a corresponding pointcloud. Make and write PLY face mesh out of it. */
int main(int ac, const char* const *av) {

    google::InitGoogleLogging(av[0]);

    float *d;
    CUDA_CHECK(cudaMalloc((void**)(&d), sizeof(float)));

    po::options_description desc("Captures RGB-D from camera. Generate and write face mesh as ply and obj");
    desc.add_options()
            ("help,H", "print this help message")
            ("model,M", po::value<std::string>(), "specify PCA model path")
            ("detector,D", po::value<std::string>(), "specify Dlib pretrained Face detection model path")
            ("graph,G", po::value<std::string>(), "specify path to PRNet graph definition")
            ("checkpoint,C", po::value<std::string>(), "specify path to pretrained PRNet checkpoint")
            ("output,O", po::value<std::string>(), "specify output PLY file path")
            ("fake,F", po::value<std::string>(), "specify directory path to captured kinect frames");
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    bool useFakeKinect = vm.count("fake") > 0;

    if(vm.count("help") > 0) {
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

    if (vm.count("output") == 0) {
        std::cout << "Please specify 'output'"  << std::endl;
        return 1;
    }


    std::string modelPath;
    std::string detectModelPath;
    std::string prnetGraphPath;
    std::string prnetChkptPath;
    std::string outputPath;
    std::string fakePath("");

    modelPath = vm["model"].as<std::string>();
    detectModelPath = vm["detector"].as<std::string>();
    prnetGraphPath = vm["graph"].as<std::string>();
    prnetChkptPath = vm["checkpoint"].as<std::string>();
    outputPath = vm["output"].as<std::string>();

    if (useFakeKinect) {

        if (vm.count("fake") == 0) {
            std::cout << "Please specify 'path' to fake device"  << std::endl;
            return 1;
        }

        fakePath = vm["fake"].as<std::string>();
    }

    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    auto imagePipe = IdentityPipe<ImageT>();
    auto cloudPipe = FastBilateralFilterPipe(); //RemoveNaNPoints();
    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>([&imagePipe](auto in)->decltype(auto){return imagePipe(in);});
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>([&cloudPipe](auto in)-> decltype(auto){return cloudPipe(in);});
    auto frontend = std::make_shared<ColorMeshPlyWriteFrontEnd>(outputPath);

    auto faceDetector = DlibFaceDetectionPipe(detectModelPath);
    auto featureDetector = PRNetFeatureDetectionPipe(fs::path(prnetGraphPath), fs::path(prnetChkptPath));
    auto lmkToScanFitting = LmkToScanRigidFittingPipe();
    PCARigidFittingPipe rigid = PCARigidFittingPipe(model);
    auto nonrigid = PCAGPUNonRigidFittingPipe();
    auto fitting2Projection = Fitting2ProjectionPipe();
    auto colorProjection = ColorProjectionPipe();

    std::shared_ptr<MorphableFaceModel> model;
    model = std::make_shared<MorphableFaceModel>(fs::path(modelPath.c_str()));

    std::shared_ptr<FittingSuitePipeMerger<ColorMesh>> merger;
    auto pipe1 = compose(faceDetector, featureDetector, lmkToScanFitting, rigid, nonrigid, fitting2Projection, colorProjection);
    merger = std::make_shared<DeviceInputPipeMerger<ColorMesh>>([&pipe1](auto in)->decltype(auto){return pipe1(in);});
    merger->addFrontEnd(frontend);

    std::shared_ptr<ImagePointCloudDevice<DeviceCloudConstT, ImageT, DeviceInputSuite, ColorMesh>> device = NULL;

    if (useFakeKinect) {
        device = std::make_shared<FakeImagePointCloudDevice <DeviceCloudConstT, ImageT, DeviceInputSuite, ColorMesh>>(fs::path(fakePath));
    } else {
        auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);
        device = std::make_shared<ImagePointCloudDeviceImpl<DeviceCloudConstT, ImageT, DeviceInputSuite, ColorMesh>>(std::move(grabber), false);
    }

    device->setCloudChannel(cloudChannel);
    device->setImageChannel(imageChannel);
    device->addMerger(merger);
    device->run();

    return 0;
}
