#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <string>

#include "align/rigid_pipe.h"
#include "align/nonrigid_pipe.h"
#include "feature/feature_detector.h"
#include "io/device.h"
#include "io/grabber.h"
#include "io/merger.h"
#include "io/frontend.h"
#include "io/ply/meshio.h"
#include "io/align/align_frontend.h"

#include "mesh/mesh.h"
#include "mesh/color_projection_pipe.h"
#include "cloud/cloud_pipe.h"


using namespace telef::feature;
using namespace telef::io;
using namespace telef::cloud;
using namespace telef::io::align;
using namespace telef::align;
using namespace telef::face;
using namespace telef::mesh;

namespace fs = std::experimental::filesystem;

/** Loads an RGB image and a corresponding pointcloud. Make and write PLY face mesh out of it. */
namespace po = boost::program_options;

int main(int ac, const char* const *av) {
    po::options_description desc("Loads an RGB image and a corresponding pointcloud. Make and write PLY face mesh out of it.");
    desc.add_options()
            ("help,H", "print help message")
            ("model,M", po::value<std::string>(), "specify PCA model path")
            ("output,O", po::value<std::string>(), "specify output PLY file path");
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if(vm.count("help") > 0 ||
       vm.count("model") == 0  || vm.count("output") == 0) {
        std::cout << desc << std::endl;
        return 1;
    }

    auto modelPath = vm["model"].as<std::string>();
    auto outputPath = vm["output"].as<std::string>();

    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);
    auto imagePipe = std::make_shared<IdentityPipe<ImageT>>();
    auto cloudPipe = std::make_shared<RemoveNaNPoints>();
    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>(std::move(imagePipe));
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>(std::move(cloudPipe));
    auto frontend = std::make_shared<ColorMeshPlyWriteFrontEnd>(outputPath);

    auto model = std::make_shared<MorphableFaceModel<150>>(fs::path(modelPath.c_str()));
    auto pipe = std::make_shared<PCARigidFittingPipe>(model)
            ->then<PCANonRigidFittingResult>(std::make_shared<PCANonRigidFittingPipe>())
            ->then<ProjectionSuite>(std::make_shared<Fitting2ProjectionPipe>())
            ->then<ColorMesh>(std::make_shared<ColorProjectionPipe>());

    auto merger = std::make_shared<FittingSuitePipeMerger<ColorMesh>>(pipe);
    merger->addFrontEnd(frontend);

    ImagePointCloudDevice<DeviceCloudConstT, ImageT, FittingSuite, ColorMesh> device {std::move(grabber)};
    device.setCloudChannel(cloudChannel);
    device.setImageChannel(imageChannel);
    device.addMerger(merger);
    device.run();

    return 0;
}