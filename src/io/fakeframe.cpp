#include "io/fakeframe.h"
#include <boost/make_shared.hpp>

namespace {
    namespace fs = std::experimental::filesystem;
}

namespace telef::io {

    FakeFrame::FakeFrame(boost::shared_ptr<DeviceCloudConstT> dc, ImagePtrT image) :
        dc(boost::make_shared<DeviceCloud>(*dc)),
        image(image) {}

    FakeFrame::FakeFrame(fs::path p) {
        dc = boost::make_shared<DeviceCloud>();
        loadDeviceCloud(p, *dc);
        image = loadPNG(p.replace_extension(".png"));
    }

    void FakeFrame::save(fs::path p) {
        saveDeviceCloud(p, *dc);
        savePNG(p.replace_extension(".png"), *image);
    }

    boost::shared_ptr<DeviceCloud> FakeFrame::getDeviceCloud() {
        return this->dc;
    }

    ImagePtrT FakeFrame::getImage() {
        return this->image;
    }
}
