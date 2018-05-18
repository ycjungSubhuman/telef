#include <pcl/io/pcd_io.h>
#include <iostream>

#include "io/devicecloud.h"
#include "io/png.h"
#include "type.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::types;
    using namespace telef::util;
    using namespace telef::io;
}

namespace telef::io {

    void saveDeviceCloud(fs::path p, const DeviceCloud &dc) {
        auto metaPath = p.replace_extension(".meta");
        auto cloudPath = p.replace_extension(".pcd");
        auto mappingPath = p.replace_extension(".mapping");
        auto imagePath = p.replace_extension(".png");

        size_t width = dc.cloud->width;
        size_t height = dc.cloud->height;
        float fx = dc.fx;
        float fy = dc.fy;

        // Write Metadata
        std::ofstream metaf(metaPath, std::ios_base::binary);
        metaf.write((char*)(&width), sizeof(size_t));
        metaf.write((char*)(&height), sizeof(size_t));
        metaf.write((char*)(&fx), sizeof(float));
        metaf.write((char*)(&fy), sizeof(float));
        metaf.close();

        // Write PointCloud
        pcl::io::savePCDFileBinary(cloudPath, *dc.cloud);

        // Write Mapping
        dc.img2cloudMapping->save(mappingPath);
    }

    void loadDeviceCloud(fs::path p, DeviceCloud &dc) {
        auto metaPath = p.replace_extension(".meta");
        auto cloudPath = p.replace_extension(".pcd");
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
        pcl::io::loadPCDFile(cloudPath, *cloud);
        cloud->width = static_cast<uint32_t>(width);
        cloud->height = static_cast<uint32_t>(height);

        // Read Mapping
        auto mapping = std::make_shared<Uv2PointIdMapT>(mappingPath);

        dc.cloud = cloud;
        dc.img2cloudMapping = mapping;
        dc.fx = fx;
        dc.fy = fy;
    }
}
