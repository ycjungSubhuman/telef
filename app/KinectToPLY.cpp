#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <boost/function.hpp>
#include <memory>
#include <chrono>
#include <thread>
#include <string>
#include <mutex>
#include <vector>


using namespace pcl;

class CloudFetcher {
private:
    using CloudConstPtr = PointCloud<PointXYZRGBA>::ConstPtr;

    std::mutex cloudMutex;
    std::string outputPath;
    CloudConstPtr currentCloud;

    std::vector<int> dummy;
    PointCloud<PointXYZRGBA> pc;

    Grabber* grabber;

    void cloudCallback(const CloudConstPtr &cloud) 
    {
        std::scoped_lock lock (cloudMutex);

        currentCloud = cloud;
    }

    void run(bool runOnce=false)
    {
        boost::function<void(const CloudConstPtr&)> _cloudCallback = boost::bind(&CloudFetcher::cloudCallback, this, _1);
        auto cloudConnection = grabber->registerCallback(_cloudCallback);
        CloudConstPtr cloud;

        grabber->start();

        while (true) {
            if(cloudMutex.try_lock())
            {
                currentCloud.swap(cloud);
                cloudMutex.unlock();
            }
            if(cloud) 
            {
                std::cout << cloud->size() << std::endl;

                removeNaNFromPointCloud(*cloud, pc, dummy);
                PLYWriter writer{};
                writer.write(outputPath, pc);

                if(runOnce) 
                {
                   break;
                }
            }
        }

        grabber->stop();
        cloudConnection.disconnect();
    }

public:
    CloudFetcher(Grabber* grabber, std::string outputPath)
        :grabber(grabber),
        outputPath(std::move(outputPath))
    {}

    void runOnce()
    {
        run(true);
    }
};

/**
 * Captures a frame from connected OpenNI2 device.
 *
 * Saves pointcloud as ply
 */

int main(int ac, char* av[]) 
{
    if(ac <= 1) 
    {
        std::cerr << "Please specify output path as the first argument(eg : ./KinnectToPLY ./wow.ply)" << std::endl;
        return 1;
    }

    std::string outputPath {av[1]};

    pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
    pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

    Grabber* grabber = new io::OpenNI2Grabber("#1", depth_mode, image_mode);
    CloudFetcher fetcher {grabber, std::move(outputPath)};

    fetcher.runOnce();

    return 0;
}

