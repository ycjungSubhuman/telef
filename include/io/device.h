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
    class ImagePointCloudDevice {
    public:
        explicit ImagePointCloudDevice(std::unique_ptr<pcl::Grabber> grabber);

        /** After added, channels will be started from the next run() */
        void addCloudChannel(std::shared_ptr<CloudChannel> channel);
        void addImageChannel(std::shared_ptr<ImageChannel> channel);

        /** Start Device and Fetch Data Through Channels
         *
         *  This call blocks thread until the grabber stops.
         */
        void run();
    private:
        std::shared_ptr<ImageChannel> imageChannel;
        std::shared_ptr<CloudChannel> cloudChannel;
        std::unique_ptr<pcl::Grabber> grabber;
    };
}