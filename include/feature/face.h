#pragma once

#include <boost/shared_ptr.hpp>
#include <cxcore.h>
#include <dlib/geometry/rectangle.h>
#include "type.h"
#include "io/merger/device_input.h"

namespace telef::feature {
    using BoundingBox = struct BoundingBox {
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

        void setBoundingBox(const dlib::rectangle& rect_) {
            x = rect_.left();
            y = rect_.top();
            width = rect_.width();
            height = rect_.height();
        }
    };

    using Feature = struct Feature {
        using Ptr = boost::shared_ptr<Feature>;
        using ConstPtr = boost::shared_ptr<const Feature>;

        // Dynamically sized Matrix is used in the case we use 2D or 3D features
        Eigen::MatrixXf points;

        // Bounding box for face
        BoundingBox boundingBox;
    };

    // TODO: Unify with FittingSuite
    using FeatureDetectSuite = struct FeatureDetectSuite {
        using Ptr = boost::shared_ptr<FeatureDetectSuite>;
        using ConstPtr = boost::shared_ptr<const FeatureDetectSuite>;

        telef::io::DeviceInputSuite::Ptr deviceInput;

        Feature::Ptr feature;
    };
}