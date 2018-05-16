#include <pcl/compression/libpng_wrapper.h>
#include <pcl/io/image.h>
#include <pcl/io/image_rgb24.h>

#include "io/png.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::types;
}

namespace telef::io {

    void savePNG(fs::path p, ImageT &image)
    {
        // Encode PNG
        std::vector<uint8_t> imageData(image.getWidth()*image.getHeight()*3);
        image.fillRaw(imageData.data());
        std::vector<uint8_t> pngData;
        pcl::io::encodeRGBImageToPNG(imageData, image.getWidth(), image.getHeight(), pngData);

        // Save PNG
        std::ofstream imagef(p, std::ios_base::binary);
        imagef.write((char*)pngData.data(), pngData.size()*sizeof(uint8_t));
        imagef.close();
    }

    ImagePtrT loadPNG(fs::path p)
    {
        std::ifstream imagef(p, std::ios_base::binary | std::ios_base::ate);
        unsigned long len = static_cast<unsigned long>(imagef.tellg());
        if(len == -1) {
            throw std::runtime_error("File does not exist");
        }
        std::vector<uint8_t> pngData(static_cast<unsigned long>(len));
        imagef.seekg(0, std::ios::beg);
        imagef.read((char*)pngData.data(), len);

        std::vector<uint8_t> imageData;

        size_t decodedWidth, decodedHeight;
        unsigned int decodedChannels;
        pcl::io::decodePNGToImage(pngData, imageData, decodedWidth, decodedHeight, decodedChannels);

        pcl::io::FrameWrapper::Ptr frame = boost::make_shared<BufferFrameWrapper>(imageData, decodedWidth, decodedHeight);
        ImagePtrT image = boost::make_shared<pcl::io::ImageRGB24>(frame);
        return image;
    }

    BufferFrameWrapper::BufferFrameWrapper(std::vector<uint8_t> data, unsigned width, unsigned height) {
        this->data = std::move(data);
        this->width = width;
        this->height = height;
    }

    const void *BufferFrameWrapper::getData() const {
        return data.data();
    }

    unsigned BufferFrameWrapper::getDataSize() const {
        return static_cast<unsigned int>(data.size());
    }

    unsigned BufferFrameWrapper::getFrameID() const {
        return 0;
    }

    unsigned BufferFrameWrapper::getHeight() const {
        return height;
    }

    pcl::uint64_t BufferFrameWrapper::getTimestamp() const {
        return 0;
    }

    unsigned BufferFrameWrapper::getWidth() const {
        return width;
    }

}