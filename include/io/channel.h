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
        using OutDataPtrT = boost::shared_ptr<OutDataT>;

        explicit Channel(std::shared_ptr<Pipe<DataT,OutDataT>> pipe) {
            this->grabberCallback = boost::bind(&Channel::_grabberCallback, this, _1);
            this->pipe = std::move(pipe);
        }
        Channel(const Channel&) = delete;

        /**
         * Called in Deveice run() Loop
         */
        OutDataPtrT onDeviceLoop() {
            std::scoped_lock lock{this->dataMutex};
            DataPtrT data;
            this->currentData.swap(data);
            if(data) {
                auto outData = this->pipe->processData(data);
                this->onOutData(outData);
                return outData;
            }
            else {
                return OutDataPtrT();
            }
        }

        /**
         * Callback to be registerd to pcl::Grabber
         */
        boost::function<void(const DataPtrT&)> grabberCallback;
    protected:

        /**
         * Handle data according to channel usage.
         *
         * This function is called before it enters any Merger.
         */
        virtual void onOutData(OutDataPtrT data) = 0;
    private:
        // allow synchronization between Grabber thread and the thread onDeviceLoop is on
        std::mutex dataMutex;

        DataPtrT currentData;
        std::shared_ptr<Pipe<DataT, OutDataT>> pipe;

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
        explicit CloudChannel(std::shared_ptr<PipeT> pipe) : Channel<CloudConstT, OutDataT>(std::move(pipe)) {}

    protected:
        void onOutData(boost::shared_ptr<OutDataT> data) override = 0;
    };

    /**
     * Fetch RGB Image from OpenNI2 Devices
     */
    template <class OutDataT>
    class ImageChannel : public Channel<ImageT, OutDataT> {
    public:
        using PipeT = Pipe<ImageT, OutDataT>;
        explicit ImageChannel(std::shared_ptr<PipeT> pipe) : Channel<ImageT, OutDataT>(std::move(pipe)) {}
    protected:
        void onOutData(boost::shared_ptr<OutDataT> data) override = 0;
    };

    template <class OutDataT>
    class DummyCloudChannel : public CloudChannel<OutDataT> {
    public:
        using PipeT = typename CloudChannel<OutDataT>::PipeT;
        explicit DummyCloudChannel(std::shared_ptr<PipeT> pipe)
                : CloudChannel<OutDataT>(std::move(pipe)) {}

    protected:
        void onOutData(boost::shared_ptr<OutDataT> data) override {
            std::cout << "CloudChannel OnData: " << data->size() << std::endl;
        }
    };

    template <class OutDataT>
    class DummyImageChannel : public ImageChannel<OutDataT> {
    public:
        using PipeT = typename ImageChannel<OutDataT>::PipeT;
        explicit DummyImageChannel(std::shared_ptr<PipeT> pipe) : ImageChannel<OutDataT>(std::move(pipe)) {}
    protected:
        void onOutData(boost::shared_ptr<OutDataT> data) override {
            std::cout << "ImageChannel OnData: ("
                      << data->getWidth()
                      << "/" << data->getHeight()
                      << ")" << std::endl;
        }
    };

}