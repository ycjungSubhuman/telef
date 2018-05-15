#pragma once

#include <experimental/filesystem>

#include "type.h"

namespace {
    using namespace telef::types;
    namespace fs = std::experimental::filesystem;
}

namespace telef::io {
    void loadImage(fs::path filepath, ImageT &target);
}
