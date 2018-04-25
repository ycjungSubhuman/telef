#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <string>

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
#include "mesh/color_projection_pipe.h"
#include "glog/logging.h"

using namespace telef::io::align;
using namespace telef::io;
using namespace telef::cloud;
using namespace telef::align;
using namespace telef::face;
using namespace telef::mesh;

namespace fs = std::experimental::filesystem;

/** Loads an RGB image and a corresponding pointcloud. Make and write PLY face mesh out of it. */
namespace po = boost::program_options;

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
std::vector<std::pair<std::string, std::vector<fs::path>>> readGroups(fs::path p) {
    std::ifstream file(p);

    std::vector<std::pair<std::string, std::vector<fs::path>>> result;

    while(!file.eof()) {
        std::string word;
        file >> word;
        if (*word.begin() == '-') // name of group
        {
            result.push_back(std::make_pair(word, std::vector<fs::path>()));
            continue;
        }
        result.back().second.push_back(fs::path(word.c_str()));
    }
    return result;
}

int main(int ac, const char* const *av) {

    google::InitGoogleLogging(av[0]);

    po::options_description desc("Captures RGB-D from camera. Generate and write face mesh as ply and obj");
    desc.add_options()
            ("help,H", "print help message")
            ("groups,K", po::value<std::string>(), "specify group file path")
            ("model,M", po::value<std::string>(), "specify PCA model path")
            ("output,O", po::value<std::string>(), "specify output PLY file path");
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if(vm.count("help") > 0) {
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("output") == 0) {
        std::cout << "Please specify 'output'"  << std::endl;
        return 1;
    }

    if (vm.count("model") == 0 && vm.count("groups") == 0) {
        std::cout << "Please specify one of 'groups' or 'model'"  << std::endl;
        return 1;
    }

    if (vm.count("groups") > 0 && vm.count("model") > 0) {
        std::cout << "Please specify only one of 'groups' and 'model'"  << std::endl;
        return 1;
    }

    std::string modelPath;
    std::string groupPath;
    std::string outputPath;

    bool isClassifyMode;
    if (vm.count("groups") > 0) {
        isClassifyMode = true;
        groupPath = vm["groups"].as<std::string>();
    }
    else {
        isClassifyMode = false;
        modelPath = vm["model"].as<std::string>();
    }

    outputPath = vm["output"].as<std::string>();

    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);
    auto imagePipe = IdentityPipe<ImageT>();
    auto cloudPipe = RemoveNaNPoints();
    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>([&imagePipe](auto in)->decltype(auto){return imagePipe(in);});
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>([&cloudPipe](auto in)-> decltype(auto){return cloudPipe(in);});
    auto frontend = std::make_shared<ColorMeshPlyWriteFrontEnd>(outputPath);

    auto nonrigid = PCANonRigidFittingPipe();
    auto fitting2Projection = Fitting2ProjectionPipe();
    auto colorProjection = ColorProjectionPipe();

    std::shared_ptr<FittingSuitePipeMerger<ColorMesh>> merger;

    if(!isClassifyMode) {
        auto model = std::make_shared<MorphableFaceModel<150>>(fs::path(modelPath.c_str()));
        auto pipe = compose(PCARigidFittingPipe(model), nonrigid, fitting2Projection, colorProjection);
        merger = std::make_shared<FittingSuitePipeMerger<ColorMesh>>([&pipe](auto in)->decltype(auto){return pipe(in);});
    }
    else {
        auto cmodel = ClassifiedMorphableModel<150>(readGroups(fs::path(groupPath.c_str())));
        auto classify = ClassifyMorphableModelPipe<150>(cmodel);
        auto crigid = ClassifiedRigidFittingPipe<150>();
        auto pipe = compose(classify, crigid, nonrigid, fitting2Projection, colorProjection);
        merger = std::make_shared<FittingSuitePipeMerger<ColorMesh>>([&pipe](auto in)->decltype(auto){return pipe(in);});
    }

    merger->addFrontEnd(frontend);

    ImagePointCloudDevice<DeviceCloudConstT, ImageT, FittingSuite, ColorMesh> device {std::move(grabber), true};
    device.setCloudChannel(cloudChannel);
    device.setImageChannel(imageChannel);
    device.addMerger(merger);
    device.run();

    return 0;
}