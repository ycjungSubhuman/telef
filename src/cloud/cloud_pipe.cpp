#include <pcl/filters/filter.h>
#include <vector>

#include "cloud/cloud_pipe.h"

using namespace telef::types;

namespace telef::cloud {
    std::unique_ptr<CloudT> RemoveNaNPoints::_processData(std::unique_ptr<CloudT> in) {
        return in;
    }
}