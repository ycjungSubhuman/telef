#pragma once

#include "io/pipe.h"
#include "cloud/cloud_pipe.h"
#include "io/channel.h"
#include "type.h"

namespace {
    using namespace telef::io;
    using namespace telef::types;
}
namespace telef::cloud {
    /** Remove NaN Positioned Points from PointCloud */
    class RemoveNaNPoints : public Pipe<DeviceCloudConstT, DeviceCloudConstT> {
    private:
        boost::shared_ptr<DeviceCloudConstT> _processData(boost::shared_ptr<DeviceCloudConstT> in) override;
    };
}
