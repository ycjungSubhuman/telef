#pragma once

#include "io/devicecloud.h"
#include "io/png.h"
#include "type.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::types;
}

namespace telef::io {
    /**
     * Fake frame that is recoreded once and played back later
     *
     * Used for feeding RGB-D dataset or running programs for same input multiple times
     **/
    class FakeFrame {
    private:
        DeviceCloudPtrT dc;
        ImagePtrT image;

    public:
        FakeFrame(DeviceCloudPtrT dc, ImagePtrT image);

        /** Load from existing file */
        FakeFrame(fs::path p);
        /** Save as a file */
        void save(fs::path p);
    };
}