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
    template <class DataT>
    class Channel{
    public:
        Channel() {
            this->grabberCallback = boost::bind(&Channel::_grabberCallback, this, _1);
        }
        Channel(const Channel&) = delete;

        /**
         * Called in Deveice run() Loop
         *
         * Acquire mutex before calling executing
         */
        void onDeviceLoop() {
            std::scoped_lock lock{this->dataMutex};
            DataT data;
            this->currentData.swap(data);
            if(data) {
                this->onData(data);
            }
        }

        /**
         * Callback to be registerd to pcl::Grabber
         */
        boost::function<void(const DataT&)> grabberCallback;
    protected:

        /**
         * Called in onDeviceLoop
         *
         * Handle data according to channel usage.
         */
        virtual void onData(DataT data) = 0;
    private:
        // allow synchronization between Grabber thread and the thread onGrabber is on
        std::mutex dataMutex;
        DataT currentData;

        void _grabberCallback(const DataT &fetchedInstance) {
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
        using DataT = pcl::PointCloud<PointT>::ConstPtr;

    protected:
        void onData(DataT data) override;
    };

    /**
     * Fetch RGB Image from OpenNI2 Devices
     */
    class ImageChannel : public Channel<pcl::io::Image::Ptr> {
    public:
        using DataT = pcl::io::Image::Ptr;
    protected:
        void onData(DataT data) override;
    };
}