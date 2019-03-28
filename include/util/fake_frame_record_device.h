#pragma once

#include <experimental/filesystem>

#include "io/device.h"
#include "io/fakeframe.h"
#include "io/merger.h"
#include "type.h"

namespace {
using namespace telef::io;
namespace fs = std::experimental::filesystem;

} // namespace

namespace telef::io {

class FakeFrameMerger
    : public SimpleBinaryMerger<ImageT, DeviceCloudConstT, FakeFrame> {
private:
  using OutPtrT = const boost::shared_ptr<FakeFrame>;
  using DataAPtrT = const boost::shared_ptr<ImageT>;
  using DataBPtrT = const boost::shared_ptr<DeviceCloudConstT>;

public:
  OutPtrT merge(DataAPtrT image, DataBPtrT dc) override;
};

/** Ready-made device object for recording fake frames */
class FakeFrameRecordDevice : public ImagePointCloudDeviceImpl<
                                  DeviceCloudConstT,
                                  ImageT,
                                  FakeFrame,
                                  FakeFrame> {
public:
  FakeFrameRecordDevice(TelefOpenNI2Grabber *grabber, fs::path recordRoot);
};
} // namespace telef::io
