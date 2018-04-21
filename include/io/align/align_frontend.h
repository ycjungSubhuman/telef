#pragma once

#include <pcl/visualization/pcl_visualizer.h>
#include "io/frontend.h"
#include "align/rigid.h"

namespace telef::io::align {
    /** Visualize Pointcloud through PCL Visualizer */

    class PCARigidVisualizerFrontEnd : public telef::io::FrontEnd<telef::align::PCARigidAlignmentSuite> {
    private:
        std::unique_ptr<vis::PCLVisualizer> visualizer;
        using InputPtrT = const boost::shared_ptr<telef::align::PCARigidAlignmentSuite>;

    public:
        void process(InputPtrT input) override;
    };
}
