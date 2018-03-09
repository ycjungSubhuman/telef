#include <pcl/filters/filter.h>
#include <vector>

#include "cloud/cloud_pipe.h"

namespace telef::cloud {
    CloudChannel::DataT RemoveNaNPoints::_processData(CloudChannel::DataT in) {
        std::vector<int> dummy;
        CloudChannel::DataT out;
        pcl::removeNaNFromPointCloud(in, out, dummy);
        return in;
    }
}