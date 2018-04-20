#pragma once

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


#include "feature/feature_detector.h"
#include "face/model.h"

using namespace telef::io;
using namespace telef::face;

namespace telef::align {
    using CloudPtrT = pcl::PointCloud<pcl::PointXYZ>::Ptr;


    // Data needed for fitting face with alignment
    //template <int ShapeRank>
    struct PCARigidAlignmentSuite {
        using Ptr = std::shared_ptr<PCARigidAlignmentSuite>;
        using ConstPtr = std::shared_ptr<const PCARigidAlignmentSuite>;

        telef::feature::FittingSuite::Ptr fittingSuite;

        // PCA
        std::shared_ptr<MorphableFaceModel<150>> pca_model;
        // Rigid alignment
        Eigen::Matrix4f transformation;
    };

    class RigidAlignment {
    public:
        virtual Eigen::Matrix4f getTransformation(CloudPtrT shape_src, const std::vector<int> &corr_src,
                                                  CloudPtrT shape_tgt, const std::vector<int> &corr_tgt) = 0;
    };

    /**
     * Face Feature Detector using IntraFace Implementation
     */
    class SVDRigidAlignment : RigidAlignment {
    public:
        Eigen::Matrix4f getTransformation(CloudPtrT shape_src, const std::vector<int> &corr_src,
                                          CloudPtrT shape_tgt, const std::vector<int> &corr_tgt) override;
    };
}
