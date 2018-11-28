#pragma once

#include <string>
#include <pcl/io/image.h>

namespace telef::io {
    void saveBMPFile(const std::string &path, unsigned char* rgb_buffer, unsigned int width, unsigned int height);
    void saveBMPFile(const std::string &path, const pcl::io::Image &image);
}