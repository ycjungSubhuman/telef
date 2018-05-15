#include <pcl/io/ply_io.h>

#include "io/devicecloud.h"
#include "type.h"

namespace fs = std::experimental::filesystem;
using telef::types;
using telef::util;

namespace telef::io {

    void saveDeviceCloud(fs::path p, const DeviceCloud &dc) {

    }

    void loadDeviceCloud(fs::path p, DeviceCloud &dc) {
        CloudPtrT cloud = boost::make_shared<CloudT>();
        pcl::io::loadPLYFile(cloudPath, *cloud);
        cloud->width =
                Uv2PointIdMapT mapping(mappingPath);
        ImagePtrT image;
    }
}
