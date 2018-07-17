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
#include "util/eigen_pcl.h"
#include "face.h"

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

            win.add_overlay(input->feature->boundingBox.getRect());
        }
    };

    /** Visualize Pointcloud through PCL Visualizer */
    class FeatureDetectFrontEnd : public FaceDetectFrontEnd {
    private:
        using InputPtrT = const boost::shared_ptr<telef::feature::FeatureDetectSuite>;

        std::unique_ptr<vis::PCLVisualizer> visualizer;

    public:

        void process(InputPtrT input) override {
                // TODO: Better combine Face and Feature detection!!!!
                // Display Face Detection
                FaceDetectFrontEnd::process(input);

                // Display Features in 3D
                auto cloud = telef::util::convert(input->feature->points);
                if (!visualizer) {
                        visualizer = std::make_unique<vis::PCLVisualizer>();
                        visualizer->setBackgroundColor(0, 0, 0);
                }

                visualizer->spinOnce();
                if(!visualizer->updatePointCloud(cloud)) {
                        visualizer->addPointCloud(cloud);
                        visualizer->setPosition (0, 0);
                        visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5);
                        //visualizer->setSize (cloud->width, cloud->height);
                        visualizer->initCameraParameters();
                }
        }
    };
}