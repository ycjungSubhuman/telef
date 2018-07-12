#pragma once

#include <pcl/visualization/pcl_visualizer.h>
#include "io/frontend.h"
#include "align/rigid.h"
#include "io/align/PCAVisualizerPrepPipe.h"

namespace telef::io::align {
    /** Visualize Pointcloud through PCL Visualizer */

    class PCAVisualizerFrontEnd : public telef::io::FrontEnd<telef::io::align::PCAVisualizerSuite> {
    private:
        std::unique_ptr<vis::PCLVisualizer> visualizer;
        using InputPtrT = const boost::shared_ptr<telef::io::align::PCAVisualizerSuite>;

    public:
        void process(InputPtrT input) override;
    };

    class PCARigidVisualizerFrontEnd : public telef::io::FrontEnd<telef::align::PCARigidAlignmentSuite> {
    private:
        std::unique_ptr<vis::PCLVisualizer> visualizer;
        using InputPtrT = const boost::shared_ptr<telef::align::PCARigidAlignmentSuite>;

    public:
        void process(InputPtrT input) override;
    };

    /** Write a colormesh */
    class ColorMeshPlyWriteFrontEnd : public FrontEnd<ColorMesh> {
    private:
        using InputPtrT = const boost::shared_ptr<ColorMesh>;
    public:
        ColorMeshPlyWriteFrontEnd(std::string outputPath) : outputPath(outputPath){}
        void process(InputPtrT input) override;
    private:
        std::string outputPath;
    };
}
