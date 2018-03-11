#pragma once

#include <vector>
#include <Eigen/Dense>
#include <pcl/io/image.h>

namespace telef::feature {

    using Feature = struct Feature {
        // Nx2 int array for points on 2d image
        Eigen::ArrayX2i points;
        // Bounding box for face
        int x;
        int y;
        int width;
        int height;
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
        using ImageT = FeatureDetector::ImageT;

        Feature getFeature(const ImageT &image) override;
    };
}