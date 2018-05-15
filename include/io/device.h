#pragma once

#include <pcl/io/grabber.h>
#include <boost/function.hpp>
#include <memory>
#include <condition_variable>
#include <experimental/filesystem>
#include <algorithm>
#include <stdio.h>

#include "io/channel.h"
#include "io/grabber.h"
#include "io/merger.h"
#include "io/frontend.h"
#include "type.h"
using namespace telef::types;

namespace {
    namespace fs = std::experimental::filesystem;
}

namespace telef::io {

    template <class CloudOutT, class ImageOutT, class MergeOutT, class MergePipeOutT>
    class ImagePointCloudDevice {
    private:
        using CloudOutPtrT = boost::shared_ptr<CloudOutT>;
        using ImageOutPtrT = boost::shared_ptr<ImageOutT>;
        using MergerT = BinaryMerger<ImageOutT, CloudOutT, MergeOutT, MergePipeOutT>;
        using FrontEndT = FrontEnd<MergePipeOutT>;
    public:
        void setCloudChannel(std::shared_ptr<CloudChannel<CloudOutT>> channel) {
            this->cloudChannel = std::move(channel);
        }
        void setImageChannel(std::shared_ptr<ImageChannel<ImageOutT>> channel) {
            this->imageChannel = std::move(channel);
        }
        void addMerger(std::shared_ptr<MergerT> merger) {
            if(!cloudChannel || !imageChannel) {
                throw std::runtime_error("Tried to add merger without either CloudChannel or ImageChannel");
            }
            this->mergers.emplace_back(merger);
        }
        virtual void run() = 0;
    protected:
        std::shared_ptr<ImageChannel<ImageOutT>> imageChannel;
        std::shared_ptr<CloudChannel<CloudOutT>> cloudChannel;
        std::vector<std::shared_ptr<MergerT>> mergers;
    };


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
    class ImagePointCloudDeviceImpl : public ImagePointCloudDevice<CloudOutT, ImageOutT, MergeOutT, MergePipeOutT> {
    private:
        using CloudOutPtrT = boost::shared_ptr<CloudOutT>;
        using ImageOutPtrT = boost::shared_ptr<ImageOutT>;
        using MergerT = BinaryMerger<ImageOutT, CloudOutT, MergeOutT, MergePipeOutT>;
        using FrontEndT = FrontEnd<MergePipeOutT>;
    public:
        explicit ImagePointCloudDeviceImpl(TelefOpenNI2Grabber *grabber, bool runOnce = false)
        :runOnce(runOnce) {
            this->grabber = grabber;

            boost::function<void(const ImagePtrT&, const boost::shared_ptr<DeviceCloud>)> callback =
                    boost::bind(&ImagePointCloudDeviceImpl::imageCloudCallback, this, _1, _2);
            boost::function<void(const ImagePtrT&)> dummyImageCallback = [](const ImagePtrT&){};
            boost::function<void(const CloudConstPtrT&)> dummyCloudCallback = [](const CloudConstPtrT&){};
            this->grabber->registerCallback(callback);

            // Register dummy callback to imageCloudCallback to work
            this->grabber->registerCallback(dummyImageCallback);
            this->grabber->registerCallback(dummyCloudCallback);
        }

        void run() {
            isRunning = true;
            grabber->start();
            this->runThread = std::thread(&ImagePointCloudDeviceImpl::_run, this);
            if(!runOnce) {
                std::cout << "Press Q-Enter to quit" << std::endl;
                while (getchar() != 'q');
            }
            std::cout << "Quitting..." << std::endl;
            isRunning = false;
            runThread.join();
            grabber->stop();
        }

    private:

        /** Start Device and Fetch Data Through Channels
         *
         *  This call blocks thread indefinitely
         */
        void _run() {
            while(isRunning) {
                CloudOutPtrT cloudOut;
                ImageOutPtrT imageOut;
                Uv2PointIdMapConstPtrT map;
                std::unique_lock<std::mutex> lk(dataMutex);
                dataCv.wait(lk);
                if (cloudChannel) {
                    cloudOut = cloudChannel->onDeviceLoop();
                    assert(cloudOut != nullptr);
                }
                if (imageChannel) {
                    imageOut = imageChannel->onDeviceLoop();
                    assert(imageOut != nullptr);
                }

                //TODO : Make passed data const to enforce consistence btw mergers
                for (const auto &m : mergers) {
                    m->run(imageOut, cloudOut);
                }
                lk.unlock();
            }
            mergers.clear();
        }


        void imageCloudCallback(const ImagePtrT &image, const boost::shared_ptr<DeviceCloud> dc) {
            std::unique_lock<std::mutex> lk(dataMutex);
            if(cloudChannel) {
                cloudChannel->grabberCallback(dc);
            }
            if(imageChannel) {
                imageChannel->grabberCallback(image);
            }
            lk.unlock();
            dataCv.notify_all();
        }

        bool runOnce;
        std::mutex dataMutex;
        std::condition_variable dataCv;
        TelefOpenNI2Grabber *grabber;
        std::thread runThread;
        volatile bool isRunning;
    };

    /** Mock class for testing */
    template <class CloudOutT, class ImageOutT, class MergeOutT, class MergePipeOutT>
    class MockImagePointCloudDevice : public ImagePointCloudDevice<CloudOutT, ImageOutT, MergeOutT, MergePipeOutT> {
    private:
        using CloudOutPtrT = boost::shared_ptr<CloudOutT>;
        using ImageOutPtrT = boost::shared_ptr<ImageOutT>;
        using MergerT = BinaryMerger<ImageOutT, CloudOutT, MergeOutT, MergePipeOutT>;
        using FrontEndT = FrontEnd<MergePipeOutT>;
    public:
        enum PlayMode {
            FIRST_FRAME_ONLY,       // Use the first frame only and terminate
            ONE_FRAME_PER_ENTER,    // Proceed to the next frame every time you press enter
            FPS_30                  // Play at 30 FPS
        };

        /**
         * Create a Mock device from previous records
         *
         * @param recordPath    recordPath is a directory that contains a list of tuples of files
         *                      (*.ply, *.mapping) eg) 1.ply, 1.mapping, 2.ply, 2.mapping ...
         *                      These records can be recorded using ImagePointCloudDevice
         */
        MockImagePointCloudDevice (fs::path recordPath, PlayMode mode=PlayMode::FPS_30) {
            this->mode = mode;
            for(int i=1; ; i++) {
                fs::path dcPath = recordPath/(fs::path(std::to_string(i)));

                auto exists = fs::exists(dcPath.replace_extension(".meta"));

                if(!exists) {
                    break;
                }

                if(mode == PlayMode::FIRST_FRAME_ONLY) {
                    break;
                }
            }
        }

        void run() override {

        }

    private:
        PlayMode mode;
        std::vector<DeviceCloud> frames;
    };
}