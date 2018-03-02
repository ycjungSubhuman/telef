#pragma once

#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <OpenNI.h>

using namespace openni;
namespace openni::face {
    class RGBDFrame 
    {
        private:
            std::shared_ptr<VideoFrameRef> depth, color;

        public:
            RGBDFrame(std::shared_ptr<VideoFrameRef> depth, std::shared_ptr<VideoFrameRef> color);
            pcl::PointCloud<pcl::PointXYZRGB> toPointCloud() const;
    };
}

