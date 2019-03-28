#pragma once

#include <opencv2/core/core.hpp>

#include <pcl/io/image.h>

/**
 * Mainly used for testing, This object bridges the OpenCV image (Mat) to
 * pcl::io::Image.
 *
 * PCL offer's no interface for loading an image from a file, this object can be
 * used to convert an Image (Mat) in OpenCV to a pcl image object
 */
namespace telef::io {
class OpencvFrameWrapper : public pcl::io::FrameWrapper {
public:
  // TODO: Add overloads to set Timestamp and FrameID
  OpencvFrameWrapper(cv::Mat metadata) : metadata_(metadata) {}

  virtual inline const void *getData() const { return (metadata_.data); }

  virtual inline unsigned getDataSize() const {
    return (metadata_.cols * metadata_.rows * metadata_.channels());
  }

  virtual inline unsigned getWidth() const { return (metadata_.cols); }

  virtual inline unsigned getHeight() const { return (metadata_.rows); }

  virtual inline unsigned getFrameID() const { return (1); }

  virtual inline uint64_t getTimestamp() const { return (1); }

  const inline cv::Mat &getMetaData() const { return (metadata_); }

private:
  cv::Mat metadata_; // Internally reference counted
};
} // namespace telef::io