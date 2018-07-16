#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/gui_widgets.h>

#include "io/frontend.h"
#include "feature/face.h"

namespace {
    using namespace telef::feature;
}

namespace telef::io {
    /** Visualize Pointcloud through PCL Visualizer */
    class FaceDetectFrontEnd : public FrontEnd<telef::feature::FeatureDetectSuite> {
    private:
        dlib::image_window win;

    public:
        using InputPtrT = const boost::shared_ptr<telef::feature::FeatureDetectSuite>;

        void process(InputPtrT input) override {
            // Convert PCL image to dlib image

            dlib::matrix<dlib::rgb_pixel> img;
            auto pclImage = input->deviceInput->rawImage;

            auto matImg = cv::Mat(pclImage->getHeight(), pclImage->getWidth(), CV_8UC3);
            pclImage->fillRGB(matImg.cols, matImg.rows, matImg.data, matImg.step);
            cv::cvtColor(matImg, matImg, CV_RGB2BGR);
            dlib::cv_image<dlib::bgr_pixel> cvImg(matImg);
            dlib::assign_image(img, cvImg);

            win.clear_overlay();
            win.set_image(img);

            dlib::rectangle rect;
            long left = input->feature->boundingBox.x;
            long top = input->feature->boundingBox.y;
            long right = left + input->feature->boundingBox.width;
            long bottom = top + input->feature->boundingBox.height;

            rect.set_bottom(bottom);
            rect.set_left(left);
            rect.set_right(right);
            rect.set_top(top);

            win.add_overlay(rect);
        }
    };
}