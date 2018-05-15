#pragma once

#include <experimental/filesystem>

#include "type.h"

namespace {
    namespace fs =
}

namespace telef::io {
    /** Point cloud with metadata */
    using DeviceCloud = struct DeviceCloud {
        pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud;
        telef::util::UvPointMapping img2cloudMapping;
        float fx;
        float fy;
    };

    void saveDeviceCloud(fs::path p, const DeviceCloud &dc);
    void loadDeviceCloud(fs::path p, DeviceCloud &dc);
}

