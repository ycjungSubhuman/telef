#pragma once

#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <OpenNI.h>

using namespace openni;
namespace openni::face {
    class RGBDFrame 
    {
        public:
            using SingleFramePtr = std::shared_ptr<VideoFrameRef>;

        private:
            SingleFramePtr depth, color;

        public:
            RGBDFrame(SingleFramePtr depth, SingleFramePtr color);
            pcl::PointCloud<pcl::PointXYZRGB> toPointCloud() const;
    };
}

