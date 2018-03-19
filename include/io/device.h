#pragma once

#include <pcl/io/grabber.h>
#include <boost/function.hpp>
#include <memory>

#include "io/channel.h"
#include "io/grabber.h"
#include "io/merger.h"
#include "io/frontend.h"
#include "type.h"
using namespace telef::types;

namespace telef::io {

    /**
     * Manages and Executes Channels for Image and PointCloud
     *
     * Interface with pcl::Grabber.
     *
     * @tparam CloudOutT Cloud Channel Piped Output Type
     * @tparam ImageOutT Image Channel Piped Output Type
     * @tparam MergeOutT Merger Output Type
     * @tparam MergePipeOutT Merger Final Output Processed by Pipe in Merger
     */
    template <class CloudOutT, class ImageOutT, class MergeOutT, class MergePipeOutT>
    class ImagePointCloudDevice {
    private:
        using CloudOutPtrT = boost::shared_ptr<CloudOutT>;
        using ImageOutPtrT = boost::shared_ptr<ImageOutT>;
        using MergerT = BinaryMerger<CloudOutT, ImageOutT, MergeOutT, MergePipeOutT>;
        using FrontEndT = FrontEnd<MergePipeOutT>;
    public:
        explicit ImagePointCloudDevice(std::unique_ptr<TelefOpenNI2Grabber> grabber) {
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
        /**
         * Add Merger to Merge CloudChannel Output and Image Channel Output into One Data
         *
         * CloudChannel and ImageChannel should be added before this being called
         * Adding merger only would not do anything useful. Add a proper frontend for it
         */
        void addImageCloudMerger(std::shared_ptr<MergerT> merger) {
            if(!cloudChannel || !imageChannel) {
                throw std::runtime_error("Tried to add merger without either CloudChannel or ImageChannel");
            }
            this->merger = merger;
        }
        /**
         * Add FrontEnd to Do Something with Side Effect Using the Output From Merger
         */
        void addFrontEnd(std::shared_ptr<FrontEndT> frontend) {
            this->frontend = frontend;
        }

        /** Start Device and Fetch Data Through Channels
         *
         *  This call blocks thread indefinitely
         */
        void run() {
            grabber->start();

            // TODO: find an appropriate program termination condition
            while (true){
                CloudOutPtrT cloudOut;
                ImageOutPtrT imageOut;

                if(cloudChannel) {
                    cloudChannel->dataMutex.lock();
                    while(!(cloudOut = cloudChannel->onDeviceLoop())) {
                        cloudChannel->dataMutex.unlock();
                        cloudChannel->dataMutex.lock();
                    }
                }
                if(imageChannel) {
                    imageChannel->dataMutex.lock();
                    while(!(imageOut = imageChannel->onDeviceLoop())) {
                        imageChannel->dataMutex.unlock();
                        imageChannel->dataMutex.lock();
                    }
                }
                if(merger && frontend) {
                    frontend->process(merger->getMergeOut(cloudOut, imageOut));
                }
                if(cloudChannel) {
                    cloudChannel->dataMutex.unlock();
                }
                if(imageChannel) {
                    imageChannel->dataMutex.unlock();
                }
            };

            grabber->stop();
        }
    private:

        void imageCloudCallback(const ImagePtrT &image, const CloudConstPtrT &cloud, const std::map<std::pair<int, int>, size_t> &uvToPointIdMap) {
            if(cloudChannel) {
                cloudChannel->grabberCallback(cloud);
            }
            if(imageChannel) {
                imageChannel->grabberCallback(image);
            }
            std::scoped_lock lock{uvToPointIdMapMutex};
            this->uvToPointIdMap = uvToPointIdMap;
        }

        std::map<std::pair<int, int>, size_t> uvToPointIdMap;
        std::mutex uvToPointIdMapMutex;
        std::shared_ptr<ImageChannel<ImageOutT>> imageChannel;
        std::shared_ptr<CloudChannel<CloudOutT>> cloudChannel;
        std::shared_ptr<MergerT> merger;
        std::shared_ptr<FrontEndT> frontend;
        std::unique_ptr<TelefOpenNI2Grabber> grabber;
    };
}