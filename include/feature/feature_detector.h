#pragma once

#include <vector>
#include <Eigen/Dense>
#include <pcl/io/image.h>
#include <cxcore.h>
#include "type.h"

using namespace telef::types;

namespace telef::feature {

    using Feature = struct Feature {
        using Ptr = std::shared_ptr<Feature>;
        using ConstPtr = std::shared_ptr<const Feature>;
        // Dynamically sized Matrix is used in the case we use 2D or 3D features
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> points;
        // Bounding box for face
        int x;
        int y;
        int width;
        int height;

        // Convenience Method
        void setBoundingBox(const cv::Rect& rect_) {
            x = rect_.x;
            y = rect_.y;
            width = rect_.width;
            height = rect_.height;
        }
    };

    // Data needed for fitting face
    using FittingSuite = struct FittingSuite {
        using Ptr = std::shared_ptr<FittingSuite>;
        using ConstPtr = std::shared_ptr<const FittingSuite>;

        Feature::Ptr landmark2d;
        CloudConstPtrT landmark3d;
        std::vector<int> invalid3dLandmarks;
        ImagePtrT rawImage;
        CloudConstPtrT rawCloud;
    };

    /**
     * Detects 2D Feature Points from 2D RGB Image
     */
    class FeatureDetector {
    public:
        // Consult http://docs.pointclouds.org/trunk/classpcl_1_1io_1_1_image.html
        using ImageT = pcl::io::Image;

        /**
         * Detects Features and Return Them
         */
        virtual Feature getFeature(const ImageT &image) = 0;
    };


    /**
     * Face Feature Detector using IntraFace Implementation
     */
    class IntraFace : FeatureDetector {
    public:
        using ImageT = FeatureDetector::ImageT;

        Feature getFeature(const ImageT &image) override;
    };
}