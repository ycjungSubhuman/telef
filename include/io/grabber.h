#pragma once

#include <pcl/io/openni2_grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/console/print.h>

using namespace pcl::io;

namespace telef::io {
    class TelefOpenNI2Grabber : public OpenNI2Grabber {
    public:
        TelefOpenNI2Grabber(const std::string &device_id, const Mode &depth_mode, const Mode &image_mode)
                : OpenNI2Grabber(device_id, depth_mode, image_mode) {
            // We don't have to delete this since ~Grabber handles them all at destruction
            image_point_cloud_rgba_signal = createSignal<sig_cb_openni_image_point_cloud_rgba>();
        }

    private:

        // Override method to achieve cloud-image synchronization
        void imageDepthImageCallback(const Image::Ptr &image, const DepthImage::Ptr &depth_image) override {
            if (point_cloud_rgb_signal_->num_slots() > 0) {
                throw std::runtime_error("Unacceptable PointXYZRGB Cloud. Use PointXYZRGBA");
            }

            if (point_cloud_rgba_signal_->num_slots() > 0 || image_point_cloud_rgba_signal->num_slots() > 0) {
                auto pc = convertToXYZRGBPointCloud<pcl::PointXYZRGBA>(image, depth_image);
                if (point_cloud_rgba_signal_->num_slots() > 0) {
                    point_cloud_rgba_signal_->operator()(pc);
                }
                if (image_point_cloud_rgba_signal->num_slots() > 0) {
                    image_point_cloud_rgba_signal->operator()(image,pc);
                }
            }

            if (image_depth_image_signal_->num_slots() > 0) {
                float reciprocalFocalLength = 1.0f / device_->getDepthFocalLength();
                image_depth_image_signal_->operator()(image, depth_image, reciprocalFocalLength);
            }
        }

        using sig_cb_openni_image_point_cloud_rgba =
        void(const boost::shared_ptr<Image> &, const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &);

        boost::signals2::signal<sig_cb_openni_image_point_cloud_rgba>* image_point_cloud_rgba_signal;
    };
}
