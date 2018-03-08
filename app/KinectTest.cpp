#include <iostream>
#include <pcl/io/openni2_grabber.h>
#include <memory>

#include "io/device.h"
#include "io/channel.h"

using namespace pcl;
using namespace telef::io;

/**
 * Try to get a single frame of point cloud and image.
 *
 * Prints size of pointcloud and width of the image
 */

int main(int ac, char* av[]) 
{
    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

    std::unique_ptr<Grabber> grabber {new io::OpenNI2Grabber("#1", depth_mode, image_mode)};

    auto imageChannel = std::make_shared<ImageChannel>();
    auto cloudChannel = std::make_shared<CloudChannel>();

    ImagePointCloudDevice device {std::move(grabber)};
    device.addCloudChannel(cloudChannel);
    device.addImageChannel(imageChannel);

    device.run();

    return 0;
}

