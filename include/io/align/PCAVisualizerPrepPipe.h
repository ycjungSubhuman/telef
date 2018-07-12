#pragma once


#include <boost/shared_ptr.hpp>
#include <memory>
#include <Eigen/Core>

#include <pcl/TextureMesh.h>

#include "io/pipe.h"
#include "align/nonrigid_pipe.h"
#include "face/model.h"
#include "mesh/mesh.h"
#include "type.h"


namespace telef::io::align {
    using PCAVisualizerSuite = struct PCAVisualizerSuite {
        pcl::TextureMesh::Ptr model;
//        CloudConstPtrT scan;

//        // Landmarks
//        vector<int> model_lmk_corr;
//        vector<int> scan_lmk_corr;
//        pcl::PointCloud<pcl::PointXYZI>::ConstPtr model_lmk;
//        pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scan_lmk;
//
//        // Geometry (closest points)
//        vector<int> model_geo_corr;
//        vector<int> scan_geo_corr;
//        pcl::PointCloud<pcl::PointXYZI>::ConstPtr geo_model;
//        pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr geo_scan;
    };

    class PCAVisualizerPrepPipe : public telef::io::Pipe<telef::align::PCANonRigidFittingResult, PCAVisualizerSuite> {
    public:
        PCAVisualizerPrepPipe(std::string outputPath)
                : telef::io::Pipe<telef::align::PCANonRigidFittingResult, PCAVisualizerSuite>(),
                        outputPath(outputPath)
        {};
    private:
        boost::shared_ptr<PCAVisualizerSuite> _processData(boost::shared_ptr<telef::align::PCANonRigidFittingResult> in) override;

        std::string outputPath;

//        int createMesh(pcl::TextureMesh &mesh, const telef::mesh::ColorMesh &colorMesh, pcl::PCLImage &tex_image);
    };
}