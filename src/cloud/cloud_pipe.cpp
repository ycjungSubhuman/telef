#include <pcl/filters/filter.h>
#include <vector>

#include "cloud/cloud_pipe.h"
#include "type.h"

using namespace telef::types;

namespace telef::cloud {
    boost::shared_ptr<MappedCloudConstT> RemoveNaNPoints::_processData(boost::shared_ptr<MappedCloudConstT> in) {
        auto cloudOut = boost::make_shared<CloudT>();
        auto cloudIn = in->first;
        auto mapping = std::make_shared<Uv2PointIdMapT>(*in->second);
        std::vector<int> mappingChange;
        pcl::removeNaNFromPointCloud(*cloudIn, *cloudOut, mappingChange);
        // out.points[i] = in.points[mappingChange[i]]
        for (const auto &p : *mapping) {
            for(unsigned long i=0; i<mappingChange.size(); i++) {
                if (p.second == mappingChange[i]) {
                    (*mapping)[p.first] = i;
                    break;
                }
            }
        }
        return boost::make_shared<MappedCloudConstT>(cloudOut, mapping);
    }
}