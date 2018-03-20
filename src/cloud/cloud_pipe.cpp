#include <pcl/filters/filter.h>
#include <vector>

#include "cloud/cloud_pipe.h"
#include "type.h"

using namespace telef::types;

namespace telef::cloud {
    boost::shared_ptr<MappedCloudConstT> RemoveNaNPoints::_processData(boost::shared_ptr<MappedCloudConstT> in) {
        auto cloudOut = boost::make_shared<CloudT>();
        auto cloudIn = in->first;
        std::vector<int> mappingChange;
        pcl::removeNaNFromPointCloud(*cloudIn, *cloudOut, mappingChange);
        in->second->updateMapping(std::move(mappingChange));
        return boost::make_shared<MappedCloudConstT>(cloudOut, in->second);
    }
}