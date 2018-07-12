#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <string>
#include <ostream>
#include <cuda_runtime_api.h>

#include "align/rigid_pipe.h"
#include "align/nonrigid_pipe.h"
#include "face/classify_pipe.h"
#include "feature/feature_detector.h"
#include "io/device.h"
#include "io/grabber.h"
#include "io/merger.h"
#include "io/frontend.h"
#include "io/ply/meshio.h"
#include "io/align/align_frontend.h"
#include "cloud/cloud_pipe.h"

#include "mesh/mesh.h"
#include "io/align/PCAVisualizerPrepPipe.h"
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
            ("help,H", "print help message")
            ("model,M", po::value<std::string>(), "specify PCA model path")
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

    if (vm.count("output") == 0) {
        std::cout << "Please specify 'output'"  << std::endl;
        return 1;
    }


    std::string modelPath;
    std::string outputPath;
    std::string fakePath("");

    modelPath = vm["model"].as<std::string>();
    outputPath = vm["output"].as<std::string>();

    if (useFakeKinect) {
        fakePath = vm["fake"].as<std::string>();
    }

    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    auto imagePipe = IdentityPipe<ImageT>();
    auto cloudPipe = RemoveNaNPoints();
    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>([&imagePipe](auto in)->decltype(auto){return imagePipe(in);});
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>([&cloudPipe](auto in)-> decltype(auto){return cloudPipe(in);});
    auto frontend = std::make_shared<PCAVisualizerFrontEnd>();

    std::shared_ptr<MorphableFaceModel<RANK>> model;
    model = std::make_shared<MorphableFaceModel<RANK>>(fs::path(modelPath.c_str()));

    PCARigidFittingPipe rigid = PCARigidFittingPipe(model);
    auto nonrigid = PCAGPUNonRigidFittingPipe();
    auto frontendPrep = PCAVisualizerPrepPipe(outputPath);

    std::shared_ptr<FittingSuitePipeMerger<PCAVisualizerSuite>> merger;
    auto pipe1 = compose(rigid, nonrigid, frontendPrep);
    merger = std::make_shared<FittingSuitePipeMerger<PCAVisualizerSuite>>([&pipe1](auto in)->decltype(auto){return pipe1(in);});
    merger->addFrontEnd(frontend);

    std::shared_ptr<ImagePointCloudDevice<DeviceCloudConstT, ImageT, FittingSuite, PCAVisualizerSuite>> device = NULL;

    if (useFakeKinect) {
        device = std::make_shared<FakeImagePointCloudDevice<
                DeviceCloudConstT,
                ImageT,
                FittingSuite,
                PCAVisualizerSuite >>(fs::path(fakePath), PlayMode::FPS_30);
    } else {
        auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);
        device = std::make_shared<ImagePointCloudDeviceImpl<
                DeviceCloudConstT,
                ImageT,
                FittingSuite,
                PCAVisualizerSuite>>(std::move(grabber), true);
    }

    device->setCloudChannel(cloudChannel);
    device->setImageChannel(imageChannel);
    device->addMerger(merger);
    device->run();

    return 0;
}
