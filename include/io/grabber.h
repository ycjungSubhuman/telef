#pragma once

#include <pcl/io/openni2_grabber.h>

#include "type.h"

namespace telef::io {

/**
 * A Modified OpenNI2Grabber in PCL library
 *
 * The main purpose of this class is to provide point cloud callbacks with
 *
 *   (Point ID) ->  (UV Coordinate) mapping
 *   Pre-defined camera focal lengthes
 *
 * These data is in DeviceCloud structure
 */
class TelefOpenNI2Grabber : public pcl::io::OpenNI2Grabber {
public:
  TelefOpenNI2Grabber(
      const std::string &device_id,
      const Mode &depth_mode,
      const Mode &image_mode);

private:
  using sig_cb_openni_image_point_cloud_rgba = void(
      const boost::shared_ptr<Image> &, const boost::shared_ptr<DeviceCloud>);

  boost::signals2::signal<sig_cb_openni_image_point_cloud_rgba>
      *image_point_cloud_rgba_signal;

  // Override method to achieve cloud-image synchronization
  void imageDepthImageCallback(
      const Image::Ptr &image, const DepthImage::Ptr &depth_image) override;

  boost::shared_ptr<DeviceCloud> mapToXYZRGBPointCloud(
      const Image::Ptr &image, const DepthImage::Ptr &depth_image);
};
} // namespace telef::io
