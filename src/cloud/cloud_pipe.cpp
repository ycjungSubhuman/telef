#include <pcl/filters/filter.h>
#include <vector>

#include "cloud/cloud_pipe.h"
#include "type.h"

using namespace telef::types;

namespace telef::cloud {
    boost::shared_ptr<CloudConstT> RemoveNaNPoints::_processData(boost::shared_ptr<CloudConstT> in) {
        auto *out = new CloudT;
        std::vector<int> dummy;
        pcl::removeNaNFromPointCloud(*in, *out, dummy);

        return boost::shared_ptr<CloudConstT>{out};
    }
}