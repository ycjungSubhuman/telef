#include "io/fakeframe.h"
#include <boost/make_shared.hpp>
#include <pcl/io/image_rgb24.h>

namespace {
namespace fs = std::experimental::filesystem;
}

namespace telef::io {

FakeFrame::FakeFrame(boost::shared_ptr<DeviceCloudConstT> dc, ImagePtrT image)
    : dc(boost::make_shared<DeviceCloud>(*dc)), image(image) {}

FakeFrame::FakeFrame(fs::path p) {
  dc = boost::make_shared<DeviceCloud>();
  loadDeviceCloud(p, *dc);
  image = loadPNG(p.replace_extension(".png"));
}

void FakeFrame::save(fs::path p) {
  saveDeviceCloud(p, *dc);
  savePNG(p.replace_extension(".png"), *image);
}

boost::shared_ptr<DeviceCloud> FakeFrame::getDeviceCloud() { return this->dc; }

namespace {
ImagePtrT CloneImage(ImagePtrT image)
{
  std::vector<uint8_t> raw(static_cast<unsigned long>(image->getDataSize()));
  image->fillRaw(raw.data());
  pcl::io::FrameWrapper::Ptr wrapper = boost::make_shared<BufferFrameWrapper>(
      raw, image->getWidth(), image->getHeight());
  ImagePtrT copy = boost::make_shared<pcl::io::ImageRGB24>(wrapper);
  return copy;
}
} // namespace

ImagePtrT FakeFrame::getImage() {
  return CloneImage(image);
}

pcl::io::DepthImage::ConstPtr FakeFrame::getDepthImage() {
  return dc->depthImage;
}
} // namespace telef::io
