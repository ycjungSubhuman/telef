#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/image.h>

namespace telef::types {
    using PointT = pcl::PointXYZRGBA;
    using CloudT = pcl::PointCloud<pcl::PointXYZRGBA>;
    using CloudConstT = const pcl::PointCloud<pcl::PointXYZRGBA>;
    using CloudPtrT = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr;
    using CloudConstPtrT = pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr;
    using ImageT = pcl::io::Image;
    using ImagePtrT = pcl::io::Image::Ptr;
    using ImageConstPtrT = pcl::io::Image::ConstPtr;

    using Uv2PointIdMapT = std::map<std::pair<int, int>, size_t>;
    using Uv2PointIdMapConstT = const Uv2PointIdMapT;
    using Uv2PointIdMapPtrT = std::shared_ptr<Uv2PointIdMapT>;
    using Uv2PointIdMapConstPtrT = std::shared_ptr<Uv2PointIdMapConstT>;
};