#pragma once

#include <vector>
#include <Eigen/Dense>
#include <pcl/io/image.h>
#include <cxcore.h>

namespace telef::feature {

    using Feature = struct Feature {
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