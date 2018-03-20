#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/image.h>
#include <unordered_map>
#include "util/UvPointMapping.h"



namespace telef::types {
    namespace hash {
        struct pairhash {
        public:
            template<typename T, typename U>
            std::size_t operator()(const std::pair<T, U> &x) const {
                //Cantor pairing function
                return (x.first+x.second)*(x.first+x.second+1)/2 + x.second;
            }
        };
    }
    using PointT = pcl::PointXYZRGBA;
    using CloudT = pcl::PointCloud<pcl::PointXYZRGBA>;
    using CloudConstT = const pcl::PointCloud<pcl::PointXYZRGBA>;
    using CloudPtrT = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr;
    using CloudConstPtrT = pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr;
    using ImageT = pcl::io::Image;
    using ImagePtrT = pcl::io::Image::Ptr;
    using ImageConstPtrT = pcl::io::Image::ConstPtr;

    using Uv2PointIdMapT = telef::util::UvPointMapping;
    using Uv2PointIdMapConstT = const Uv2PointIdMapT;
    using Uv2PointIdMapPtrT = std::shared_ptr<Uv2PointIdMapT>;
    using Uv2PointIdMapConstPtrT = std::shared_ptr<Uv2PointIdMapConstT>;

    using MappedCloudT = std::pair<CloudConstPtrT, Uv2PointIdMapPtrT>;
    using MappedCloudConstT = const MappedCloudT;
    using MappedCloudPtrT = std::shared_ptr<MappedCloudT>;
    using MappedCloudConstPtrT = std::shared_ptr<MappedCloudConstT>;
};