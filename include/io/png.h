#pragma once

#include <experimental/filesystem>
#include <pcl/io/image.h>

#include "type.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::types;
}

namespace telef::io {

    class BufferFrameWrapper : public pcl::io::FrameWrapper {
    public:
        BufferFrameWrapper(std::vector<uint8_t> data, unsigned width, unsigned height);
        const void* getData() const override;
        unsigned getDataSize() const override;
        unsigned getFrameID() const override;
        unsigned getHeight() const override;
        pcl::uint64_t getTimestamp() const override;
        unsigned getWidth() const override;
    private:
        std::vector<uint8_t> data;
        unsigned width;
        unsigned height;
    };

    void savePNG(fs::path p, ImageT &image);
    ImagePtrT loadPNG(fs::path p);
}