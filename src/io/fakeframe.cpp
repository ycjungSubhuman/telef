#include "io/fakeframe.h"

namespace {
    namespace fs = std::experimental::filesystem;
}

namespace telef::io {

    FakeFrame::FakeFrame(std::shared_ptr<DeviceCloud> dc, ImagePtrT image) :
        dc(dc),
        image(image) {}

    FakeFrame::FakeFrame(fs::path p) {
        dc = std::make_shared<DeviceCloud>();
        loadDeviceCloud(p, *dc);
        image = loadPNG(p.replace_extension(".png"));
    }

    void FakeFrame::save(fs::path p) {
        saveDeviceCloud(p, *dc);
        savePNG(p.replace_extension(".png"), *image);
    }
}