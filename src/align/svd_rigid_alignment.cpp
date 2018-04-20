#include "feature/feature_detector.h"

#include <iostream>
#include <string>

#include <pcl/registration/transformation_estimation_svd.h>

#include "align/rigid.h"

using namespace std;

namespace telef::align {

    Eigen::Matrix4f SVDRigidAlignment::getTransformation(CloudPtrT shape_src, const std::vector<int> &corr_src,
                                                         CloudPtrT shape_tgt, const std::vector<int> &corr_tgt) {
        Eigen::Matrix4f transformation;
        pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
        svd.estimateRigidTransformation(*shape_src, corr_src, *shape_tgt, corr_tgt, transformation);
    }


}