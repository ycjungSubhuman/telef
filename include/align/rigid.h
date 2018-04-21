#pragma once

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "feature/feature_detector.h"
//Problem including face/model.h due to meshio and cluster due to modelio name conflicts??
//#include "face/model.h"

//Forward Declare MorphableFaceModel
namespace telef
{
    namespace face
    {
        template <int ShapeRank> class MorphableFaceModel;
    }
}

namespace telef::align {

    // Data needed for fitting face with alignment
    using PCARigidAlignmentSuite =//template <int ShapeRank>
    struct PCARigidAlignmentSuite {
        using Ptr = std::shared_ptr<PCARigidAlignmentSuite>;
        using ConstPtr = std::shared_ptr<const PCARigidAlignmentSuite>;

        boost::shared_ptr<telef::feature::FittingSuite> fittingSuite;

        // PCA
        std::shared_ptr<telef::face::MorphableFaceModel<150>> pca_model;
        // Rigid alignment
        Eigen::Matrix4f transformation;
    };
}
