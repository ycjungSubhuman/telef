#pragma once

#include "io/pipe.h"
#include "cloud/cloud_pipe.h"
#include "io/channel.h"
#include "type.h"

using namespace telef::io;
using namespace telef::types;
namespace telef::cloud {
    /** Remove NaN Positioned Points from PointCloud */
    class RemoveNaNPoints : public Pipe<CloudT, CloudT> {
    private:
        std::unique_ptr<CloudT> _processData(std::unique_ptr<CloudT> in) override;
    };
}
