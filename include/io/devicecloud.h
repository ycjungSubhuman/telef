#pragma once

#include <experimental/filesystem>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "util/UvPointMapping.h"

namespace {
    namespace fs = std::experimental::filesystem;
}

namespace telef::io {
    /**
     * Point cloud with metadata.
     *
     * Used as the first input of ImagePointCloudDevice
     * Also, used as a frame in recoding in MockImagePointCLoudDevice.
     *
     **/
    using DeviceCloud = struct DeviceCloud {
        pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud;
        std::shared_ptr<telef::util::UvPointMapping> img2cloudMapping;
        float fx;
        float fy;
    };

    void saveDeviceCloud(fs::path p, const DeviceCloud &dc);
    void loadDeviceCloud(fs::path p, DeviceCloud &dc);
}

