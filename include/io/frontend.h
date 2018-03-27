#pragma once

#include <iostream>
#include <sstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/png_io.h>
#include <pcl/io/ply_io.h>
#include <experimental/filesystem>
#include <ctime>
#include <future>
#include "type.h"
#include "camera.h"
#include "bmp.h"

using namespace telef::types;
using namespace telef::feature;
namespace vis = pcl::visualization;

namespace telef::io {
    /**
     * Do Something with Side Effect Provided InputT
     *
     * Used as final step in Device
     */
    template<class InputT>
    class FrontEnd {
    private:
        using InputPtrT = const boost::shared_ptr<InputT>;
    public:
        virtual ~FrontEnd() = default;
        virtual void process(InputPtrT input)=0;
    };

    class DummyCloudFrontEnd : public FrontEnd<CloudConstT> {
    private:
        using InputPtrT = const CloudConstPtrT;
    public:
        void process(InputPtrT input) override {
            std::cout << "DummyCloudFrontEnd : " << input->size() << std::endl;
        }
    };

    /** Visualize Pointcloud through PCL Visualizer */
    class Landmark3DVisualizerFrontEnd : public FrontEnd<FittingSuite> {
    private:
        std::unique_ptr<vis::PCLVisualizer> visualizer;
        using InputPtrT = const boost::shared_ptr<FittingSuite>;

    public:
        void process(InputPtrT input) override {
            auto cloud = input->landmark3d;
            if (!visualizer) {
                visualizer = std::make_unique<vis::PCLVisualizer>();
                visualizer->setBackgroundColor(0, 0, 0);
            }
            visualizer->spinOnce();
            if(!visualizer->updatePointCloud(cloud)) {
                visualizer->addPointCloud(cloud);
                visualizer->setPosition (0, 0);
                visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5);
                visualizer->setSize (cloud->width, cloud->height);
                visualizer->initCameraParameters();
            }
        }
    };

    /** Export received points in Pointcloud as csv */
    class FittingSuiteWriterFrontEnd : public FrontEnd<FittingSuite> {
    private:
        using InputPtrT = const boost::shared_ptr<FittingSuite>;
        bool ignoreIncomplete;
        bool saveRGB;
        bool saveRawCloud;
        int expectedPointsCount;
        int frameCount;
        std::experimental::filesystem::path folder;
    public:
        explicit FittingSuiteWriterFrontEnd (bool ignoreIncomplete=true, bool saveRGB=false, bool saveRawCloud=false, int expectedPointsCount=49):
                ignoreIncomplete(ignoreIncomplete),
                saveRGB(saveRGB),
                saveRawCloud(saveRawCloud),
                expectedPointsCount(expectedPointsCount),
                frameCount(0) {
            std::time_t t= std::time(nullptr);
            auto localTime = std::localtime(&t);
            std::stringstream folderNameStream;
            folderNameStream << "capture_" << localTime->tm_mday << "-"
                       << localTime->tm_mon+1 << "-"
                       << localTime->tm_year+1900 << "-"
                       << localTime->tm_hour << "-"
                       << localTime->tm_min << "-"
                       << localTime->tm_sec;
            folder=folderNameStream.str();
            std::experimental::filesystem::create_directory(folder);
        }
        void process(InputPtrT input) override {
            if(ignoreIncomplete && (input->landmark3d->size() != expectedPointsCount)) {
                return;
            }

            auto save = std::bind(&FittingSuiteWriterFrontEnd::_save, this, std::placeholders::_1);
            std::async(save, input);

        }

        void _save(InputPtrT input) {
            // point order is preserved from 0 to 48(see LandMarkMerger::merge)
            // So it is safe to just put every points in order
            std::stringstream sstream;
            for(const auto &p :input->landmark3d->points) {
                sstream << p.x << ",";
                sstream << p.y << ",";
                sstream << p.z << "\n";
            }

            auto pathPrefix = folder/("frame_" + std::to_string(frameCount));
            std::ofstream outCsv;
            outCsv.open(pathPrefix.replace_extension(std::string(".csv")));
            outCsv << sstream.str();
            outCsv.close();
            if(saveRGB) {
                telef::io::saveBMPFile(pathPrefix.replace_extension(std::string(".bmp")), *input->rawImage);
            }
            if(saveRawCloud) {
                pcl::io::savePLYFile(pathPrefix.replace_extension(std::string(".ply")), *input->rawCloud);
            }

            frameCount++;
            std::cout << "Captured" << std::endl;
        }
    };
}
