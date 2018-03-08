#include <utility>
#include "io/device.h"
#include <chrono>
#include <thread>

namespace telef::io {
    ImagePointCloudDevice::ImagePointCloudDevice(std::unique_ptr<pcl::Grabber> grabber) {
        this->grabber = std::move(grabber);
    }

    void ImagePointCloudDevice::addCloudChannel(std::shared_ptr<CloudChannel> channel) {
        this->cloudChannel = std::move(channel);
    }

    void ImagePointCloudDevice::addImageChannel(std::shared_ptr<ImageChannel> channel) {
        this->imageChannel = std::move(channel);
    }

    void ImagePointCloudDevice::run() {
        if(cloudChannel) {
            grabber->registerCallback(cloudChannel->grabberCallback);
        }
        if(imageChannel) {
            grabber->registerCallback(cloudChannel->grabberCallback);
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
}
