#pragma once

#include <pcl/io/grabber.h>
#include <boost/function.hpp>
#include <memory>

#include "io/channel.h"

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
            if(cloudChannel) {
                grabber->registerCallback(cloudChannel->grabberCallback);
            }
            if(imageChannel) {
                grabber->registerCallback(imageChannel->grabberCallback);
            }
            grabber->start();

            while (true)
            {
                if(cloudChannel)
                {
                    cloudChannel->onDeviceLoop();
                }
                if(imageChannel)
                {
                    imageChannel->onDeviceLoop();
                }
            }

            grabber->stop();
        }
    private:
        std::shared_ptr<ImageChannel<ImageOutT>> imageChannel;
        std::shared_ptr<CloudChannel<CloudOutT>> cloudChannel;
        std::unique_ptr<pcl::Grabber> grabber;
    };
}