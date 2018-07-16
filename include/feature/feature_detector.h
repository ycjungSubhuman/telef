#pragma once

#include <vector>
#include <Eigen/Dense>
#include <pcl/io/image.h>
#include <cxcore.h>
#include "type.h"
#include "io/merger/device_input.h"
#include "feature/face.h"

namespace {
    using namespace telef::types;
}

namespace telef::feature {

    // Data needed for fitting face
    // TODO: Refactor, Unify with feature/face.h and add to own header file
    using FittingSuite = struct FittingSuite {
        using Ptr = std::shared_ptr<FittingSuite>;
        using ConstPtr = std::shared_ptr<const FittingSuite>;

        Feature::Ptr landmark2d;
        CloudConstPtrT landmark3d;
        std::vector<int> invalid3dLandmarks;
        ImagePtrT rawImage;
        CloudConstPtrT rawCloud;
        std::vector<int> rawCloudLmkIdx;
        float fx;
        float fy;
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