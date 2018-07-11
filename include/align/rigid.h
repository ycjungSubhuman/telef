#pragma once

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "feature/feature_detector.h"
#include "type.h"
#include "face/model.h"

namespace telef::align {

    // Data needed for fitting face with alignment
    using PCARigidAlignmentSuite =//template <int ShapeRank>
    struct PCARigidAlignmentSuite {
        using Ptr = std::shared_ptr<PCARigidAlignmentSuite>;
        using ConstPtr = std::shared_ptr<const PCARigidAlignmentSuite>;

        boost::shared_ptr<telef::feature::FittingSuite> fittingSuite;

        // PCA
        std::shared_ptr<telef::face::MorphableFaceModel<SHAPE_RANK>> pca_model;
        // Rigid alignment
        Eigen::Matrix4f transformation;
        telef::types::ImagePtrT image;
        CloudConstPtrT rawCloud;

        float fx;
        float fy;
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}
