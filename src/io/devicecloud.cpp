#include <pcl/io/ply_io.h>
#include <pcl/compression/libpng_wrapper.h>
#include <iostream>
#include <unitypes.h>

#include "io/devicecloud.h"
#include "type.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::types;
    using namespace telef::util;
}

namespace telef::io {

    void saveDeviceCloud(fs::path p, const DeviceCloud &dc) {
        //TODO : FInish

    }

    void loadDeviceCloud(fs::path p, DeviceCloud &dc) {
        auto metaPath = p.replace_extension(".meta");
        auto cloudPath = p.replace_extension(".ply");
        auto mappingPath = p.replace_extension(".mapping");
        auto imagePath = p.replace_extension(".png");

        size_t width;
        size_t height;
        float fx;
        float fy;

        // Read Metadata
        std::ifstream metaf(metaPath, std::ios_base::binary);
        metaf.read((char*)(&width), sizeof(size_t));
        metaf.read((char*)(&height), sizeof(size_t));
        metaf.read((char*)(&fx), sizeof(float));
        metaf.read((char*)(&fy), sizeof(float));
        metaf.close();

        // Read PointCloud
        CloudPtrT cloud = boost::make_shared<CloudT>();
        pcl::io::loadPLYFile(cloudPath, *cloud);
        cloud->width = static_cast<uint32_t>(width);
        cloud->height = static_cast<uint32_t>(height);

        // Read Mapping
        Uv2PointIdMapT mapping(mappingPath);

        // Read Image
        //TODO : FInish
    }
}
