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

int main(int ac, const char* const *av) {

    google::InitGoogleLogging(av[0]);

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
    auto imagePipe = IdentityPipe<ImageT>();
    auto cloudPipe = RemoveNaNPoints();
    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>([&imagePipe](auto in)->decltype(auto){return imagePipe(in);});
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>([&cloudPipe](auto in)-> decltype(auto){return cloudPipe(in);});
    auto frontend = std::make_shared<ColorMeshPlyWriteFrontEnd>(outputPath);

    auto model = std::make_shared<MorphableFaceModel<150>>(fs::path(modelPath.c_str()));
    auto rigid = PCARigidFittingPipe(model);
    auto nonrigid = PCANonRigidFittingPipe();
    auto fitting2Projection = Fitting2ProjectionPipe();
    auto colorProjection = ColorProjectionPipe();
    auto pipe = compose(rigid, nonrigid, fitting2Projection, colorProjection);

    auto merger = std::make_shared<FittingSuitePipeMerger<ColorMesh>>([&pipe](auto in)->decltype(auto){return pipe(in);});
    merger->addFrontEnd(frontend);

    ImagePointCloudDevice<DeviceCloudConstT, ImageT, FittingSuite, ColorMesh> device {std::move(grabber), true};
    device.setCloudChannel(cloudChannel);
    device.setImageChannel(imageChannel);
    device.addMerger(merger);
    device.run();

    return 0;
}