#pragma once

#include <iostream>
#include <sstream>
#include <ctime>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <experimental/filesystem>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/png_io.h>
#include <pcl/io/ply_io.h>

#include "type.h"
#include "camera.h"
#include "bmp.h"
#include "feature/feature_detector.h"
#include "io/fakeframe.h"

namespace {
    using namespace telef::types;
    using namespace telef::feature;
    namespace vis = pcl::visualization;
}

namespace telef::io {
    /**
     * Do Something with Side Effect Provided InputT
     *
     * Used as final step in Device
     */
    template<class InputT>
    class FrontEnd {
    public:
        using InputPtrT = const boost::shared_ptr<InputT>;
        virtual ~FrontEnd() = default;
        virtual void process(InputPtrT input)=0;
    };

    class DummyCloudFrontEnd : public FrontEnd<CloudConstT> {
    public:
        using InputPtrT = const CloudConstPtrT;
        void process(InputPtrT input) override {
            std::cout << "DummyCloudFrontEnd : " << input->size() << std::endl;
        }
    };

    /** Visualize Pointcloud through PCL Visualizer */
    class Landmark3DVisualizerFrontEnd : public FrontEnd<FittingSuite> {
    private:
        std::unique_ptr<vis::PCLVisualizer> visualizer;

    public:
        using InputPtrT = const boost::shared_ptr<FittingSuite>;
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

    /**
     * Frontend with asynchronous processing logic
     *
     * A call to process() will return immediately. The computation is queued and processed FIFO way.
     **/
    template<class T>
    class AsyncFrontEnd : public FrontEnd<T> {
    public:
        using InputPtrT = const boost::shared_ptr<T>;
    private:

        void jobLoop() {
            while(isJobGranted) {
                std::unique_lock ul{dataMutex};
                nonempty.wait(ul);
                InputPtrT data = pendingData.front();
                pendingData.pop();
                ul.unlock();

                _process(data);
            }
            std::cout << "Finishing up Job..." << std::endl;
            while(!pendingData.empty()) {
                std::cout << pendingData.size() << "frames to go" << std::endl;
                _process(pendingData.front());
                pendingData.pop();
            }
            std::cout << "Job Complete" << std::endl;
        }

        virtual void _process(InputPtrT input)=0;

        // Check if to queue up the data
        // If any input does not require processing, return false
        virtual bool checkProcessNeeded(InputPtrT input) {
            return true;
        }

        std::mutex dataMutex;
        std::condition_variable nonempty;
        std::queue<boost::shared_ptr<T>> pendingData;
        std::thread jobThread;
        volatile bool isJobGranted; // Controlled by the thread 'process()' is on.

    public:
        AsyncFrontEnd():
                jobThread(std::thread(&AsyncFrontEnd::jobLoop, this)),
                isJobGranted(true)
        {}
        virtual ~AsyncFrontEnd() {
            isJobGranted = false;
            nonempty.notify_all();
            jobThread.join();
        }
        virtual void process(InputPtrT input) {
            if(checkProcessNeeded(input)) {
                std::unique_lock<std::mutex> ul{dataMutex};
                pendingData.push(input);
                ul.unlock();
                nonempty.notify_all();
            }
        }
    };

    /** Export received points in Pointcloud as csv */
    class FittingSuiteWriterFrontEnd : public AsyncFrontEnd<FittingSuite> {
    public:
        using InputPtrT = const boost::shared_ptr<FittingSuite>;
    private:
        bool ignoreIncomplete;
        bool saveRGB;
        bool saveRawCloud;
        int expectedPointsCount;
        int frameCount;

        std::experimental::filesystem::path folder;

        bool checkProcessNeeded(InputPtrT input) override;

        void _process(InputPtrT input) override;

    public:
        explicit FittingSuiteWriterFrontEnd (bool ignoreIncomplete=true, bool saveRGB=false,
                                             bool saveRawCloud=false, int expectedPointsCount=49);
    };

    /** Record Fake Frames */
    class RecordFakeFrameFrontEnd : public AsyncFrontEnd<FakeFrame> {
    public:
        using InputPtrT = const boost::shared_ptr<FakeFrame>;
        RecordFakeFrameFrontEnd(fs::path recordRoot);

        void _process(InputPtrT input) override;
    private:
        fs::path recordRoot;
        int frameCount;
    };
}
