#pragma once

#include <iostream>
#include <sstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <experimental/filesystem>
#include <ctime>
#include "type.h"
#include "camera.h"

using namespace telef::types;
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
    class CloudVisualizerFrontEnd : public FrontEnd<CloudConstT> {
    private:
        std::unique_ptr<vis::PCLVisualizer> visualizer;
        using InputPtrT = const CloudConstPtrT;

    public:
        void process(InputPtrT input) override {
            if (!visualizer) {
                visualizer = std::make_unique<vis::PCLVisualizer>();
                visualizer->setBackgroundColor(0, 0, 0);
            }
            visualizer->spinOnce();
            if(!visualizer->updatePointCloud(input)) {
                visualizer->addPointCloud(input);
                visualizer->setPosition (0, 0);
                visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5);
                visualizer->setSize (input->width, input->height);
                visualizer->initCameraParameters();
            }
        }
    };

    /** Export received points in Pointcloud as csv */
    class Point3DCsvWriterFrontEnd : public FrontEnd<CloudConstT> {
    private:
        using InputPtrT = const CloudConstPtrT;
        bool ignoreIncomplete;
        int expectedPointsCount;
        int frameCount;
        std::experimental::filesystem::path folder;
    public:
        explicit Point3DCsvWriterFrontEnd (bool ignoreIncomplete=true, int expectedPointsCount=49):
                ignoreIncomplete(ignoreIncomplete),
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
            if(input->size() != expectedPointsCount) {
                return;
            }

            // point order is preserved from 0 to 48(see LandMarkMerger::merge)
            // So it is safe to just put every points in order
            std::stringstream sstream;
            for(const auto &p :input->points) {
                sstream << p.x << ",";
                sstream << p.y << ",";
                sstream << p.z << "\n";
            }

            std::ofstream outFile;
            outFile.open(folder/("frame_" + std::to_string(frameCount)));
            frameCount++;
            std::cout << "Captured" << std::endl;
            outFile << sstream.str();
            outFile.close();
        }
    };
}
