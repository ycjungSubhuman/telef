#pragma once

#include <assert.h>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace telef::util {

// Convert Eigen::Vector into pcl::Pointcloud<PointXYZ>
// TODO: make generic to PointT?? to handle PointXYZ
//    inline pcl::PointCloud<pcl::PointXYZ>::Ptr convert(const Eigen::VectorXf&
//    vector) {
//
//        assert(("Vector doesn't contains (x,y,z) for each n 3D vertex",
//        vector.size() % 3 == 0));
//
//        // TODO: Make util vector(xyz,..) to pointcloud
//        pcl::PointCloud<pcl::PointXYZ>::Ptr ptCld(new
//        pcl::PointCloud<pcl::PointXYZ>());
//
//        for (int idx = 0; idx < vector.size(); idx+=3) {
//            ptCld->points.emplace_back(vector[idx],
//                                       vector[idx+1],
//                                       vector[idx+2]);
//        }
//        ptCld->width = (int)ptCld->points.size();
//        ptCld->height = 1;
//
//        return ptCld;
//    }

inline pcl::PointCloud<pcl::PointXYZ>::Ptr
convert(const Eigen::MatrixXf &matrix) {
  assert(
      ("Matrix doesn't contains (x,y,z) for each n 3D vertex",
       matrix.size() % 3 == 0));

  // TODO: Make util vector(xyz,..) to pointcloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr ptCld(
      new pcl::PointCloud<pcl::PointXYZ>());

  for (int idx = 0; idx < matrix.size(); idx += 3) {
    ptCld->points.emplace_back(
        matrix.data()[idx], matrix.data()[idx + 1], matrix.data()[idx + 2]);
  }
  ptCld->width = (int)ptCld->points.size();
  ptCld->height = 1;

  return ptCld;
}
} // namespace telef::util
