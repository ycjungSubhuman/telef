#include <cmath>
#include <map>
#include <pcl/console/print.h>
#include <pcl/point_cloud.h>
#include <tuple>

#include "io/grabber.h"

using namespace pcl::io;
using namespace telef::types;

namespace {
// Treat color as chars, float32, or uint32
typedef union {
  struct {
    unsigned char Blue;
    unsigned char Green;
    unsigned char Red;
    unsigned char Alpha;
  };
  float float_value;
  uint32_t long_value;
} RGBValue;
} // namespace

namespace telef::io {
TelefOpenNI2Grabber::TelefOpenNI2Grabber(
    const std::string &device_id,
    const Mode &depth_mode,
    const Mode &image_mode)
    : OpenNI2Grabber(device_id, depth_mode, image_mode) {
  // We don't have to delete this since ~Grabber handles them all at destruction
  image_point_cloud_rgba_signal =
      createSignal<sig_cb_openni_image_point_cloud_rgba>();
}

boost::shared_ptr<DeviceCloud> TelefOpenNI2Grabber::mapToXYZRGBPointCloud(
    const Image::Ptr &image, const DepthImage::Ptr &depth_image) {
  boost::shared_ptr<pcl::PointCloud<PointT>> cloud(new pcl::PointCloud<PointT>);
  auto uvToPointIdMap =
      std::make_shared<Uv2PointIdMapT>(image->getWidth(), image->getHeight());

  cloud->header.seq = depth_image->getFrameID();
  cloud->header.stamp = depth_image->getTimestamp();
  cloud->header.frame_id = rgb_frame_id_;
  cloud->height = std::max(image_height_, depth_height_);
  cloud->width = std::max(image_width_, depth_width_);
  cloud->is_dense = false;

  cloud->points.resize(cloud->height * cloud->width);

  // Generate default camera parameters
  float fx = device_->getDepthFocalLength(); // Horizontal focal length
  float fy = device_->getDepthFocalLength(); // Vertcal focal length
  float cx = ((float)depth_width_ - 1.f) / 2.f; // Center x
  float cy = ((float)depth_height_ - 1.f) / 2.f; // Center y

  // Load pre-calibrated camera parameters if they exist
  if (std::isfinite(depth_parameters_.focal_length_x))
    fx = static_cast<float>(depth_parameters_.focal_length_x);

  if (std::isfinite(depth_parameters_.focal_length_y))
    fy = static_cast<float>(depth_parameters_.focal_length_y);

  if (std::isfinite(depth_parameters_.principal_point_x))
    cx = static_cast<float>(depth_parameters_.principal_point_x);

  if (std::isfinite(depth_parameters_.principal_point_y))
    cy = static_cast<float>(depth_parameters_.principal_point_y);

  // Get inverse focal length for calculations below
  float fx_inv = 1.0f / fx;
  float fy_inv = 1.0f / fy;

  const uint16_t *depth_map = (const uint16_t *)depth_image->getData();
  if (depth_image->getWidth() != depth_width_ ||
      depth_image->getHeight() != depth_height_) {
    // Resize the image if nessacery
    depth_resize_buffer_.resize(depth_width_ * depth_height_);
    depth_map = depth_resize_buffer_.data();
    depth_image->fillDepthImageRaw(
        depth_width_, depth_height_, (unsigned short *)depth_map);
  }

  const uint8_t *rgb_buffer = (const uint8_t *)image->getData();
  if (image->getWidth() != image_width_ ||
      image->getHeight() != image_height_) {
    // Resize the image if nessacery
    color_resize_buffer_.resize(image_width_ * image_height_ * 3);
    rgb_buffer = color_resize_buffer_.data();
    image->fillRGB(
        image_width_,
        image_height_,
        (unsigned char *)rgb_buffer,
        image_width_ * 3);
  }

  float bad_point = std::numeric_limits<float>::quiet_NaN();

  // set xyz to Nan and rgb to 0 (black)
  if (image_width_ != depth_width_) {
    PointT pt;
    pt.x = pt.y = pt.z = bad_point;
    pt.b = pt.g = pt.r = 0;
    pt.a = 255; // point has no color info -> alpha = max => transparent
    cloud->points.assign(cloud->points.size(), pt);
  }

  // fill in XYZ values
  unsigned step = cloud->width / depth_width_;
  unsigned skip = cloud->width - (depth_width_ * step);

  int value_idx = 0;
  int point_idx = 0;
  for (int v = 0; v < depth_height_; ++v, point_idx += skip) {
    for (int u = 0; u < depth_width_; ++u, ++value_idx, point_idx += step) {
      PointT &pt = cloud->points[point_idx];
      /// @todo Different values for these cases
      // Check for invalid measurements

      OniDepthPixel pixel = depth_map[value_idx];
      if (pixel != 0 && pixel != depth_image->getNoSampleValue() &&
          pixel != depth_image->getShadowValue()) {
        uvToPointIdMap->addSingle(u, v, (size_t)point_idx);
        pt.z = depth_map[value_idx] * 0.001f; // millimeters to meters
        pt.x = (static_cast<float>(u) - cx) * pt.z * fx_inv;
        pt.y = (static_cast<float>(v) - cy) * pt.z * fy_inv;
      } else {
        pt.x = pt.y = pt.z = bad_point;
      }
    }
  }

  // fill in the RGB values
  step = cloud->width / image_width_;
  skip = cloud->width - (image_width_ * step);

  value_idx = 0;
  point_idx = 0;
  RGBValue color;
  color.Alpha = 0xff;

  for (unsigned yIdx = 0; yIdx < image_height_; ++yIdx, point_idx += skip) {
    for (unsigned xIdx = 0; xIdx < image_width_;
         ++xIdx, point_idx += step, value_idx += 3) {
      PointT &pt = cloud->points[point_idx];

      color.Red = rgb_buffer[value_idx];
      color.Green = rgb_buffer[value_idx + 1];
      color.Blue = rgb_buffer[value_idx + 2];

      pt.rgba = color.long_value;
    }
  }
  cloud->sensor_origin_.setZero();
  cloud->sensor_orientation_.setIdentity();

  auto result = boost::make_shared<DeviceCloud>();
  result->cloud = cloud;
  result->img2cloudMapping = uvToPointIdMap;
  result->fx = fx;
  result->fy = fy;
  result->depthImage = depth_image;

  return result;
}

void TelefOpenNI2Grabber::imageDepthImageCallback(
    const Image::Ptr &image, const DepthImage::Ptr &depth_image) {
  if (point_cloud_rgb_signal_->num_slots() > 0) {
    throw std::runtime_error(
        "Unacceptable PointXYZRGB Cloud. Use PointXYZRGBA");
  }

  if (point_cloud_rgba_signal_->num_slots() > 0 ||
      image_point_cloud_rgba_signal->num_slots() > 0) {
    auto deviceCloud = mapToXYZRGBPointCloud(image, depth_image);
    if (point_cloud_rgba_signal_->num_slots() > 0) {
      point_cloud_rgba_signal_->operator()(deviceCloud->cloud);
    }
    if (image_point_cloud_rgba_signal->num_slots() > 0) {
      image_point_cloud_rgba_signal->operator()(image, deviceCloud);
    }
  }

  if (image_depth_image_signal_->num_slots() > 0) {
    float reciprocalFocalLength = 1.0f / device_->getDepthFocalLength();
    image_depth_image_signal_->operator()(
        image, depth_image, reciprocalFocalLength);
  }
}

} // namespace telef::io
