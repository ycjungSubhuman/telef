#pragma once

#include <pcl/io/grabber.h>
#include <boost/function.hpp>
#include <chrono>
#include <memory>
#include <queue>
#include <condition_variable>
#include <experimental/filesystem>
#include <algorithm>
#include <stdio.h>

#include "io/channel.h"
#include "io/grabber.h"
#include "io/merger.h"
#include "io/frontend.h"
#include "io/fakeframe.h"
#include "type.h"

namespace {
    using namespace telef::types;
    namespace fs = std::experimental::filesystem;
    using namespace std::chrono_literals;
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
                if (this->cloudChannel) {
                    cloudOut = this->cloudChannel->onDeviceLoop();
                    assert(cloudOut != nullptr);
                }
                if (this->imageChannel) {
                    imageOut = this->imageChannel->onDeviceLoop();
                    assert(imageOut != nullptr);
                }

                //TODO : Make passed data const to enforce consistence btw this->mergers
                for (const auto &m : this->mergers) {
                    m->run(imageOut, cloudOut);
                }
                lk.unlock();
            }
            this->mergers.clear();
        }


        void imageCloudCallback(const ImagePtrT &image, const boost::shared_ptr<DeviceCloud> dc) {
            std::unique_lock<std::mutex> lk(dataMutex);
            if(this->cloudChannel) {
                this->cloudChannel->grabberCallback(dc);
            }
            if(this->imageChannel) {
                this->imageChannel->grabberCallback(image);
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

    enum PlayMode {
        FIRST_FRAME_ONLY,       // Use the first frame only and terminate
        ONE_FRAME_PER_ENTER,    // Proceed to the next frame every time you press enter
        FPS_30,                  // Play at 30 FPS
        FPS_30_LOOP             // Play at 30 FPS + Loop
    };

    /** Fake device for easy experiments */
    template <class CloudOutT, class ImageOutT, class MergeOutT, class MergePipeOutT>
    class FakeImagePointCloudDevice : public ImagePointCloudDevice<CloudOutT, ImageOutT, MergeOutT, MergePipeOutT> {
    private:
        using CloudOutPtrT = boost::shared_ptr<CloudOutT>;
        using ImageOutPtrT = boost::shared_ptr<ImageOutT>;
        using MergerT = BinaryMerger<ImageOutT, CloudOutT, MergeOutT, MergePipeOutT>;
        using FrontEndT = FrontEnd<MergePipeOutT>;
    public:

        /**
         * Create a Fake device from previous records
         *
         * @param recordPath    recordPath is a directory that contains a list of tuples of files
         *                      (*.ply, *.meta, *mapping, *.png)
         *                      These records can be recorded using FakeFrameRecordDevice
         */
        FakeImagePointCloudDevice (fs::path recordPath, PlayMode mode=PlayMode::FIRST_FRAME_ONLY) {
            std::cout << "Loading Fake Frames..." << std::endl;
            this->mode = mode;
            for(int i=1; ; i++) {
                fs::path framePath = recordPath/(fs::path(std::to_string(i)));

                auto exists = fs::exists(framePath.replace_extension(".pcd"));

                if(!exists) {
                    break;
                }

                auto frame = std::make_shared<FakeFrame>(framePath);
                frames.push(frame);
                std::cout << "Loaded frame " << std::to_string(i) << std::endl;

                if(mode == PlayMode::FIRST_FRAME_ONLY) {
                    break;
                }
            }
        }

        void run() override {
            while (!frames.empty()) {
                auto frameProcessStartTime = std::chrono::system_clock::now();
                auto frame = frames.front();
                frames.pop();

                this->cloudChannel->grabberCallback(frame->getDeviceCloud());
                this->imageChannel->grabberCallback(frame->getImage());

                auto cloudOut = this->cloudChannel->onDeviceLoop();
                auto imageOut = this->imageChannel->onDeviceLoop();

                for(auto &m : this->mergers) {
                    m->run(imageOut, cloudOut);
                }
                auto frameProcessEndTime = std::chrono::system_clock::now();

                auto frameProcessTime = frameProcessEndTime - frameProcessStartTime;

                if(mode==PlayMode::FIRST_FRAME_ONLY) {
                    break;
                }
                else if (mode==PlayMode::ONE_FRAME_PER_ENTER) {
                    std::cout << "Press Enter to proceed to the next frame.." << std::endl;
                    getchar();
                }
                else if (mode==PlayMode::FPS_30 || mode==PlayMode::FPS_30_LOOP) {
                    auto sleepTime = 33ms - frameProcessTime;
                    if (sleepTime.count() >= 0) {
                        std::this_thread::sleep_for(sleepTime);
                    }

                    if(mode == PlayMode::FPS_30_LOOP) {
                        frames.push(frame);
                    }
                }
            }
        }

    private:
        PlayMode mode;
        std::queue<std::shared_ptr<FakeFrame>> frames;
    };
}
