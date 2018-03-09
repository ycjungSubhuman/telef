#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <mutex>
#include <boost/function.hpp>
#include <pcl/io/image.h>

namespace telef::io {
    /**
     * Data Channel for Device.
     */
    template <class DataPtrT>
    class Channel{
    public:
        Channel() {
            this->grabberCallback = boost::bind(&Channel::_grabberCallback, this, _1);
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
                this->onData(data);
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

        void _grabberCallback(const DataPtrT &fetchedInstance) {
            std::scoped_lock lock{this->dataMutex};
            this->currentData = fetchedInstance;
        }
    };



    /**
     * Fetch XYZRGBA Point Cloud from OpenNI2 Devices
     */
    class CloudChannel : public Channel<pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr> {
    private:
        using PointT = pcl::PointXYZRGBA;
    public:
        using DataT = pcl::PointCloud<PointT>;
        using DataPtrT = DataT::ConstPtr;

    protected:
        void onData(DataPtrT data) override;
    };

    /**
     * Fetch RGB Image from OpenNI2 Devices
     */
    class ImageChannel : public Channel<pcl::io::Image::Ptr> {
    public:
        using DataT = pcl::io::Image;
        using DataPtrT = DataT::Ptr;
    protected:
        void onData(DataPtrT data) override;
    };
}