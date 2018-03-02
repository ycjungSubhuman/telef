#include "RGBDFrame.h"
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <OpenNI.h>

using namespace openni;
namespace openni::face {

    RGBDFrame::RGBDFrame(std::shared_ptr<VideoFrameRef> depth, std::shared_ptr<VideoFrameRef> color)
    {
        this->depth = depth;
        this->color = color;
    }

    pcl::PointCloud<pcl::PointXYZRGB> RGBDFrame::toPointCloud() const
    {
        std::cout << "Stub toPointCloud" << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        return cloud;
    }
}

