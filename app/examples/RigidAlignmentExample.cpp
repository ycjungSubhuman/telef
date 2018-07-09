#include <iostream>
#include <pcl/io/openni2_grabber.h>
//#include <boost/program_options.hpp>

#include "cloud/cloud_pipe.h"
#include "io/device.h"
#include "io/align/align_frontend.h"
#include "align/rigid_pipe.h"
//#include "align/rigid.h"
#include "face/model.h"

namespace {
    using namespace telef::io;
    using namespace telef::types;
    using namespace telef::cloud;
    using namespace telef::feature;
}

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

    auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);

    auto imagePipe = IdentityPipe<ImageT>();
    auto cloudPipe = RemoveNaNPoints();

    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>([&imagePipe](auto in)->decltype(auto){return imagePipe(in);});
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>([&cloudPipe](auto in)->decltype(auto){return cloudPipe(in);});


    auto model = std::make_shared<telef::face::MorphableFaceModel<RANK>>(fs::path("../pcamodels/example"));
    auto rigidFitPipe = telef::align::PCARigidFittingPipe(model);

    //auto rigidFitPipe = std::make_shared<telef::align::PCARigidFittingPipe>();
    auto merger = std::make_shared<FittingSuitePipeMerger<telef::align::PCARigidAlignmentSuite>>([&rigidFitPipe](auto in)->decltype(auto){return rigidFitPipe(in);});


    //auto merger = std::make_shared<RigidAlignFrontEnd>();

//    auto csvFrontend = std::make_shared<FittingSuiteWriterFrontEnd>(vm.count("include-holes") == 0,
//                                                                    vm.count("save-rgb") > 0,
//                                                                    vm.count("save-cloud") > 0);
//    merger->addFrontEnd(csvFrontend);
//    if (vm.count("view")) {
//        auto viewFrontend = std::make_shared<RigidModelVisualizerFrontEnd>();
//        merger->addFrontEnd(viewFrontend);
//    }

    auto viewFrontend = std::make_shared<telef::io::align::PCARigidVisualizerFrontEnd>();
    merger->addFrontEnd(viewFrontend);

    ImagePointCloudDeviceImpl<DeviceCloudConstT, ImageT,
            FittingSuite, telef::align::PCARigidAlignmentSuite> device{std::move(grabber)};

    device.setCloudChannel(cloudChannel);
    device.setImageChannel(imageChannel);
    device.addMerger(merger);

    device.run();

    return 0;
}
