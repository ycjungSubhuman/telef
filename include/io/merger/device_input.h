#pragma once

#include <boost/shared_ptr.hpp>
#include "type.h"

namespace {
    using namespace telef::types;
}

namespace telef::io {


    using DeviceInputSuite = struct DeviceInputSuite {
        using Ptr = boost::shared_ptr<DeviceInputSuite>;
        using ConstPtr = boost::shared_ptr<const DeviceInputSuite>;

        ImagePtrT rawImage;
        CloudConstPtrT rawCloud;
        std::shared_ptr<telef::util::UvPointMapping> img2cloudMapping;
        float fx;
        float fy;
    };
}