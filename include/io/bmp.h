#pragma once

#include <pcl/io/image.h>
#include <string>

namespace telef::io {
void saveBMPFile(const std::string &path, const pcl::io::Image &image);
}