#pragma once

#include "io/pipe.h"

using namespace telef::io;
namespace telef::cloud {
    /** Remove NaN Positioned Points from PointCloud */
    class RemoveNaNPoints : public Pipe<CloudChannel::DataT, CloudChannel::DataT> {
    private:
        CloudChannel::DataT _processData(CloudChannel::DataT in) override;
    };
}
