#include <iostream>
#include <memory>

#include "util/fake_frame_record_device.h"

namespace {
using namespace telef::io;
namespace fs = std::experimental::filesystem;
} // namespace

int main(int argc, char **argv) {
  if (argc <= 1) {
    std::cout << "Usage : $ RecordFakeFrame <record root path>" << std::endl;
    return 1;
  }

  std::string arg(argv[1]);
  fs::path recordRoot(arg);
  if (!fs::create_directory(recordRoot)) {
    std::cerr << "Failed to create fresh directory" << std::endl;
  }

  pcl::io::OpenNI2Grabber::Mode depth_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  pcl::io::OpenNI2Grabber::Mode image_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);

  auto device = FakeFrameRecordDevice(grabber, recordRoot);

  device.run();

  return 0;
}
