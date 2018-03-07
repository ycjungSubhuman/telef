#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <mutex>

namespace telef::io {
    /**
     * Objects that Fetch Some Data from Devices On Regular Basis
     */
    template <class T>
    class Fetcher {
    public:
        /**
         * Callback to be called when instance data is ready
         *
         * While this function runs, Grabber thread cannot access the fetchedInstance.
         */
        virtual void onDataReady(const T &fetchedInstance) = 0;

        /**
         * Callback to be called by pcl::Grabber thread.
         *
         * Don't call this directly, pass this function to pcl::io::Grabber::registerCallback
         */
        void callback(const T &fetchedInstance);
    private:
        // allow synchronization between Grabber thread and the thread onDataReady is on
        std::mutex dataMutex;
    };



    /**
     * Fetch XYZRGBA Point Cloud from OpenNI2 Devices
     */
    class CloudFetcher : Fetcher<pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr> {
    public:
        using PointT = pcl::PointXYZRGBA;
        using InstancePtr = pcl::PointCloud<PointT>::ConstPtr;

        void onDataReady(const InstancePtr &fetchedInstance) override;
    private:
        InstancePtr currentCloud;
    };
}