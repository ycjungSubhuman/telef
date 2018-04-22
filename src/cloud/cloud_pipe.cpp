#include <pcl/filters/filter.h>
#include <vector>

#include "cloud/cloud_pipe.h"
#include "type.h"

using namespace telef::types;

namespace telef::cloud {

    RemoveNaNPoints::RemoveNaNPoints() {
        this->composed = std::bind(&RemoveNaNPoints::_processData, this, std::placeholders::_1);
    }

    boost::shared_ptr<DeviceCloudConstT> RemoveNaNPoints::_processData(boost::shared_ptr<DeviceCloudConstT> in) {
        auto cloudOut = boost::make_shared<CloudT>();
        auto cloudIn = in->cloud;
        std::vector<int> mappingChange;
        pcl::removeNaNFromPointCloud(*cloudIn, *cloudOut, mappingChange);
        in->img2cloudMapping->updateMapping(std::move(mappingChange));

        auto result = boost::make_shared<DeviceCloudT>();
        result->img2cloudMapping = in->img2cloudMapping;
        result->cloud = cloudOut;
        result->fx = in->fx;
        result->fy = in->fy;
        return result;
    }
}