#pragma once

#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
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

    class CloudVisualizerFrontEnd : public FrontEnd<CloudConstT> {
    private:
        std::unique_ptr<vis::PCLVisualizer> visualizer;
        using InputPtrT = const CloudConstPtrT;

    public:
        CloudVisualizerFrontEnd() {
        }

        void process(InputPtrT input) override {
            if (!visualizer) {
                visualizer = std::make_unique<vis::PCLVisualizer>();
                visualizer->setBackgroundColor(0, 0, 0);
            }
            visualizer->spinOnce();
            if(!visualizer->updatePointCloud(input)) {
                visualizer->addPointCloud(input);
                visualizer->setPosition (0, 0);
                visualizer->setSize (input->width, input->height);
                visualizer->setPointCloudRenderingProperties(vis::PCL_VISUALIZER_POINT_SIZE, 1);
                visualizer->initCameraParameters();
                //visualizer->setCameraFieldOfView(NUI_CAMERA_COLOR_NOMINAL_VERTICAL_FOV);
            }
        }
    };
}
