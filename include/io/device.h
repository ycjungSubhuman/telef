#pragma once

#include <pcl/io/grabber.h>
#include <boost/function.hpp>
#include <memory>
#include <condition_variable>
#include <algorithm>
#include <stdio.h>

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
        using MergerT = BinaryMerger<ImageOutT, CloudOutT, MergeOutT, MergePipeOutT>;
        using FrontEndT = FrontEnd<MergePipeOutT>;
    public:
        explicit ImagePointCloudDevice(std::unique_ptr<TelefOpenNI2Grabber> grabber) {
            this->grabber = std::move(grabber);

            boost::function<void(const ImagePtrT&, const CloudConstPtrT&, const Uv2PointIdMapPtrT&)> callback =
                    boost::bind(&ImagePointCloudDevice::imageCloudCallback, this, _1, _2, _3);
            boost::function<void(const ImagePtrT&)> dummyImageCallback = [](const ImagePtrT&){};
            boost::function<void(const CloudConstPtrT&)> dummyCloudCallback = [](const CloudConstPtrT&){};
            this->grabber->registerCallback(callback);

            // Register dummy callback to imageCloudCallback to work
            this->grabber->registerCallback(dummyImageCallback);
            this->grabber->registerCallback(dummyCloudCallback);
        }

        /** After added, channels will be started from the next run() */
        void setCloudChannel(std::shared_ptr<CloudChannel<CloudOutT>> channel) {
            this->cloudChannel = std::move(channel);
        }
        void setImageChannel(std::shared_ptr<ImageChannel<ImageOutT>> channel) {
            this->imageChannel = std::move(channel);
        }

        /**
         * Add Merger to Merge CloudChannel Output and Image Channel Output into One Data
         *
         * CloudChannel and ImageChannel should be added before this being called
         */
        void addMerger(std::shared_ptr<MergerT> merger) {
            if(!cloudChannel || !imageChannel) {
                throw std::runtime_error("Tried to add merger without either CloudChannel or ImageChannel");
            }
            this->mergers.emplace_back(merger);
        }
        void run() {
            std::cout << "Press Q-Enter to quit" << std::endl;
            isRunning = true;
            grabber->start();
            this->runThread = std::thread(&ImagePointCloudDevice::_run, this);
            while(getchar()!='q');
            std::cout << "Quitting..." << std::endl;
            isRunning = false;
            runThread.join();

            //grabber->stop();
        }

    private:

        /** Start Device and Fetch Data Through Channels
         *
         *  This call blocks thread indefinitely
         */
        void _run() {
            while (isRunning){
                CloudOutPtrT cloudOut;
                ImageOutPtrT imageOut;
                Uv2PointIdMapConstPtrT map;
                std::unique_lock<std::mutex> lk(dataMutex);
                dataCv.wait(lk);
                if(cloudChannel) {
                    cloudOut = cloudChannel->onDeviceLoop();
                    assert(cloudOut != nullptr);
                }
                if(imageChannel) {
                    imageOut = imageChannel->onDeviceLoop();
                    assert(imageOut != nullptr);
                }

                //TODO : Make passed data const to enforce consistence btw mergers
                for(const auto &m : mergers) {
                    m->run(imageOut, cloudOut);
                }
                lk.unlock();
            };
            mergers.clear();
        }



        void imageCloudCallback(const ImagePtrT &image, const CloudConstPtrT &cloud, const Uv2PointIdMapPtrT &uvToPointIdMap) {
            std::unique_lock<std::mutex> lk(dataMutex);
            if(cloudChannel) {
                cloudChannel->grabberCallback(boost::make_shared<MappedCloudT>(cloud, uvToPointIdMap));
            }
            if(imageChannel) {
                imageChannel->grabberCallback(image);
            }
            lk.unlock();
            dataCv.notify_all();
        }

        std::mutex dataMutex;
        std::condition_variable dataCv;
        std::shared_ptr<ImageChannel<ImageOutT>> imageChannel;
        std::shared_ptr<CloudChannel<CloudOutT>> cloudChannel;
        std::vector<std::shared_ptr<MergerT>> mergers;
        std::unique_ptr<TelefOpenNI2Grabber> grabber;
        std::thread runThread;
        volatile bool isRunning;
    };
}