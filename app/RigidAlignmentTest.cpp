#include <iostream>
#include <pcl/io/openni2_grabber.h>
//#include <boost/program_options.hpp>

#include "io/device.h"
#include "cloud/cloud_pipe.h"
#include "image/image_pipe.h"
#include "align/rigid.h"
#include "align/rigid_pipe.h"
#include "io/align/align_frontend.h"

using namespace telef::io;
using namespace telef::types;
using namespace telef::cloud;
using namespace telef::image;
using namespace telef::feature;
using namespace telef::align;

/**
 * Project IntraFace landmark points onto captured pointcloud and Rigid Fit a PCA model
 */

//namespace po = boost::program_options;

//int main(int ac, const char* const * av) {
//    po::options_description desc("Capture 3D Landmark Points from RGBD Camera and Save into multiple CSV files");
//    desc.add_options()
//            ("help,H", "print help message")
//            ("include-holes,I", "include landmark projections with holes")
//            ("view,V", "open GUI window for monitoring captured landmarks")
//            ("save-rgb,R", "save corresponding RGB images too")
//            ("save-cloud,C", "save corresponding raw cloud images too");
//
//    po::variables_map vm;
//    po::store(po::parse_command_line(ac, av, desc), vm);
//    po::notify(vm);
//
//    if (vm.count("help")) {
//        std::cout << desc << std::endl;
//        return 1;
//    }

int main(int argc, char** argv) {
    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

    auto grabber = std::make_unique<TelefOpenNI2Grabber>("#1", depth_mode, image_mode);

    auto imagePipe = std::make_shared<IdentityPipe<ImageT>>();
    auto cloudPipe = std::make_shared<RemoveNaNPoints>();

    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>(std::move(imagePipe));
    auto cloudChannel = std::make_shared<DummyCloudChannel<MappedCloudConstT>>(std::move(cloudPipe));

    //telef::face::MorphableFaceModel<150> model(fs::path("../pcamodels/example"));


//    auto rigidFitPipe = std::make_shared<PCARigidFittingPipe>(model);
    auto rigidFitPipe = std::make_shared<PCARigidFittingPipe>();
    auto merger = std::make_shared<FittingSuitePipeMerger<PCARigidAlignmentSuite>>(rigidFitPipe);


    //auto merger = std::make_shared<RigidAlignFrontEnd>();

//    auto csvFrontend = std::make_shared<FittingSuiteWriterFrontEnd>(vm.count("include-holes") == 0,
//                                                                    vm.count("save-rgb") > 0,
//                                                                    vm.count("save-cloud") > 0);
//    merger->addFrontEnd(csvFrontend);
//    if (vm.count("view")) {
//        auto viewFrontend = std::make_shared<RigidModelVisualizerFrontEnd>();
//        merger->addFrontEnd(viewFrontend);
//    }

    auto viewFrontend = std::make_shared<align::PCARigidVisualizerFrontEnd>();
    merger->addFrontEnd(viewFrontend);

    ImagePointCloudDevice<MappedCloudConstT, ImageT, FittingSuite, PCARigidAlignmentSuite> device{std::move(grabber)};
    device.setCloudChannel(cloudChannel);
    device.setImageChannel(imageChannel);
    device.addMerger(merger);

    device.run();

    return 0;
}