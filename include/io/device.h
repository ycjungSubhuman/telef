#pragma once

#include <pcl/io/grabber.h>
#include <boost/function.hpp>
#include <memory>

#include "io/channel.h"
#include "type.h"
using namespace telef::types;

namespace telef::io {

    /**
     * Manages and Executes Channels for Image and PointCloud
     *
     * Interface with pcl::Grabber.
     */
    template <class CloudOutT, class ImageOutT>
    class ImagePointCloudDevice {
    public:
        explicit ImagePointCloudDevice(std::unique_ptr<pcl::Grabber> grabber) {
            this->grabber = std::move(grabber);

            boost::function<void(const ImagePtrT&, const CloudConstPtrT&)> callback =
                    boost::bind(&ImagePointCloudDevice::imageCloudCallback, this, _1, _2);
            boost::function<void(const ImagePtrT&)> dummyImageCallback = [](const ImagePtrT&){};
            boost::function<void(const CloudConstPtrT&)> dummyCloudCallback = [](const CloudConstPtrT&){};
            this->grabber->registerCallback(callback);

            // Register dummy callback to imageCloudCallback to work
            this->grabber->registerCallback(dummyImageCallback);
            this->grabber->registerCallback(dummyCloudCallback);
        }

        /** After added, channels will be started from the next run() */
        void addCloudChannel(std::shared_ptr<CloudChannel<CloudOutT>> channel) {
            this->cloudChannel = std::move(channel);
        }
        void addImageChannel(std::shared_ptr<ImageChannel<ImageOutT>> channel) {
            this->imageChannel = std::move(channel);
        }

        /** Start Device and Fetch Data Through Channels
         *
         *  This call blocks thread until the grabber stops.
         */
        void run() {
            grabber->start();

            while (true)
            {
                if(cloudChannel) {
                    cloudChannel->onDeviceLoop();
                }
                if(imageChannel) {
                    imageChannel->onDeviceLoop();
                }
            }

            grabber->stop();
        }
    private:
        void imageCloudCallback(const ImagePtrT &image, const CloudConstPtrT &cloud) {
            if(cloudChannel) {
                cloudChannel->grabberCallback(cloud);
            }
            if(imageChannel) {
                imageChannel->grabberCallback(image);
            }
        }

        std::shared_ptr<ImageChannel<ImageOutT>> imageChannel;
        std::shared_ptr<CloudChannel<CloudOutT>> cloudChannel;
        std::unique_ptr<pcl::Grabber> grabber;
    };
}