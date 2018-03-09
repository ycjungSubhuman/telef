#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <mutex>
#include <boost/function.hpp>
#include <pcl/io/image.h>
#include <memory>

#include "io/pipe.h"
#include "type.h"

using namespace telef::types;

namespace telef::io {
    /**
     * Data Channel for Device.
     */
    template <class DataT, class OutDataT>
    class Channel{
    public:
        // Use boost shared_ptr for pcl compatibility
        using DataPtrT = boost::shared_ptr<DataT>;

        explicit Channel(std::unique_ptr<Pipe<DataT,OutDataT>> pipe) {
            this->grabberCallback = boost::bind(&Channel::_grabberCallback, this, _1);
            this->pipe = std::move(pipe);
        }
        Channel(const Channel&) = delete;

        /**
         * Called in Deveice run() Loop
         */
        void onDeviceLoop() {
            std::scoped_lock lock{this->dataMutex};
            DataPtrT data;
            this->currentData.swap(data);
            if(data) {
                this->onData(this->pipe->processData(data));
            }
        }

        /**
         * Callback to be registerd to pcl::Grabber
         */
        boost::function<void(const DataPtrT&)> grabberCallback;
    protected:

        /**
         * Handle data according to channel usage.
         */
        virtual void onData(DataPtrT data) = 0;
    private:
        // allow synchronization between Grabber thread and the thread onDeviceLoop is on
        std::mutex dataMutex;
        DataPtrT currentData;
        std::unique_ptr<Pipe<DataT, OutDataT>> pipe;

        void _grabberCallback(const DataPtrT &fetchedInstance) {
            std::scoped_lock lock{this->dataMutex};
            this->currentData = fetchedInstance;
        }
    };

    /**
     * Fetch XYZRGBA Point Cloud from OpenNI2 Devices
     */
    template <class OutDataT>
    class CloudChannel : public Channel<CloudConstT, OutDataT> {
    public:
        using PipeT = Pipe<CloudConstT, OutDataT>;
        explicit CloudChannel(std::unique_ptr<PipeT> pipe) : Channel<CloudConstT, OutDataT>(std::move(pipe)) {}

    protected:
        void onData(CloudConstPtrT data) override {
            std::cout << "CloudChannel OnData: " << data->size() << std::endl;
        }
    };

    /**
     * Fetch RGB Image from OpenNI2 Devices
     */
    template <class OutDataT>
    class ImageChannel : public Channel<ImageT, OutDataT> {
    public:
        using PipeT = Pipe<ImageT, OutDataT>;
        explicit ImageChannel(std::unique_ptr<PipeT> pipe) : Channel<ImageT, OutDataT>(std::move(pipe)) {}
    protected:
        void onData(ImagePtrT data) override {
            std::cout << "ImageChannel OnData: ("
                      << data->getWidth()
                      << "/" << data->getHeight()
                      << ")" << std::endl;
        }
    };
}