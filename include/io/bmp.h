#pragma once

#include <string>
#include <pcl/io/image.h>

namespace telef::io {
    void saveBMPFile(const std::string &path, const pcl::io::Image &image);
}