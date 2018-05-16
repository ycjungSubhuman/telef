#include <pcl/filters/filter.h>
#include <vector>

#include "cloud/cloud_pipe.h"
#include "type.h"

using namespace telef::types;

namespace telef::cloud {
    boost::shared_ptr<DeviceCloudConstT> RemoveNaNPoints::_processData(boost::shared_ptr<DeviceCloudConstT> in) {
        CloudT cloudOut(*in->cloud);
        auto cloudIn = in->cloud;
        std::vector<int> mappingChange;
        pcl::removeNaNFromPointCloud(*cloudIn, cloudOut, mappingChange);

        auto result = boost::make_shared<DeviceCloudT>();
        result->img2cloudMapping = in->img2cloudMapping;
        result->img2cloudMapping->updateMapping(std::move(mappingChange));
        result->cloud = boost::make_shared<CloudConstT>(cloudOut);
        result->fx = in->fx;
        result->fy = in->fy;
        return result;
    }
}
