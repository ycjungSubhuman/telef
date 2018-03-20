#include <iostream>
#include <pcl/io/openni2_grabber.h>

#include "io/device.h"
#include "cloud/cloud_pipe.h"
#include "image/image_pipe.h"

using namespace telef::io;
using namespace telef::types;
using namespace telef::cloud;
using namespace telef::image;
using namespace telef::feature;

/**
 * Project IntraFace landmark points onto captured pointcloud
 */

int main(int ac, char* av[])
{
    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

    auto grabber = std::make_unique<TelefOpenNI2Grabber>("#1", depth_mode, image_mode);

    auto imagePipe = std::make_shared<IdentityPipe<ImageT>>();
    auto cloudPipe = std::make_shared<IdentityPipe<MappedCloudConstT>>();

    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>(std::move(imagePipe));
    auto cloudChannel = std::make_shared<DummyCloudChannel<MappedCloudConstT>>(std::move(cloudPipe));

    auto merger = std::make_shared<LandmarkMerger>();
    std::shared_ptr<FrontEnd<CloudConstT>> frontend;
    if (ac > 1 && strcmp(av[1], "csv")==0) {
        frontend = std::make_shared<Point3DCsvWriterFrontEnd>();
    }
    else {
        frontend = std::make_shared<CloudVisualizerFrontEnd>();
    }

    ImagePointCloudDevice<MappedCloudConstT, ImageT, CloudConstT, CloudConstT> device {std::move(grabber)};
    device.addCloudChannel(cloudChannel);
    device.addImageChannel(imageChannel);
    device.addImageCloudMerger(merger);
    device.addFrontEnd(frontend);

    device.run();

    return 0;
}

