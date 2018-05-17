#include "io/channel.h"
#include "cloud/cloud_pipe.h"
#include "io/frontend.h"
#include "util/fake_frame_record_device.h"

namespace {
    using namespace telef::io;
    using namespace telef::cloud;
    using namespace telef::types;
}

namespace telef::io {

    FakeFrameMerger::OutPtrT FakeFrameMerger::merge(FakeFrameMerger::DataAPtrT image, FakeFrameMerger::DataBPtrT dc) {
        return boost::make_shared<FakeFrame>(dc, image);
    }

    FakeFrameRecordDevice::FakeFrameRecordDevice(TelefOpenNI2Grabber *grabber, fs::path recordRoot) :
        ImagePointCloudDeviceImpl<DeviceCloudConstT, ImageT, FakeFrame, FakeFrame>(grabber, false)
    {
        auto removeNaN = std::make_shared<RemoveNaNPoints>();
        this->cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>(
                [removeNaN](auto in)->decltype(auto){return (*removeNaN)(in);}
                );
        this->imageChannel = std::make_shared<DummyImageChannel<ImageT>>();
        auto merger = std::make_shared<FakeFrameMerger>();
        auto frontend = std::make_shared<RecordFakeFrameFrontEnd>(recordRoot);
        merger->addFrontEnd(frontend);
        this->mergers.push_back(merger);
    }
}