#pragma once

#include "util/UvPointMapping.h"
#include <experimental/filesystem>
#include <pcl/io/image_depth.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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
  pcl::io::DepthImage::ConstPtr depthImage;
  float fx;
  float fy;
};

void saveDeviceCloud(fs::path p, const DeviceCloud &dc);
void loadDeviceCloud(fs::path p, DeviceCloud &dc);
} // namespace telef::io
