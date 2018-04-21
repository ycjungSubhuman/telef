
#include "io/align/align_frontend.h"

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include "pcl/common/transforms.h"

#include "io/frontend.h"
#include "util/eigen_pcl.h"
#include "face/model.h"

using namespace telef::align;
using namespace telef::io;
using namespace telef::face;

namespace telef::io::align {


    void PCARigidVisualizerFrontEnd::process(InputPtrT input) {
        std::cout << "In PCARigidVisualizerFrontEnd::process\n";
        auto lmksPtCld = input->fittingSuite->landmark3d;

        ColorMesh meanMesh = input->pca_model->genMesh(Eigen::VectorXf::Zero(150));
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcaPtCld = telef::util::convert(meanMesh.position);

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*pcaPtCld, *transformed_cloud, input->transformation);

//        pcl::PCLPointCloud2 mesh;
//        pcl::toPCLPointCloud2(pcaPointCloud, mesh);
//        pcl::PolygonMesh polyMesh;
//        polyMesh.cloud = mesh;
//        polyMesh.polygons[];
//
//        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> single_color(lmksPtCld, 255, 0, 0);

        if (!visualizer) {
            visualizer = std::make_unique<vis::PCLVisualizer>();
            visualizer->setBackgroundColor(0, 0, 0);
        }
        visualizer->spinOnce();
        if (!visualizer->updatePointCloud(lmksPtCld) && !visualizer->updatePointCloud(pcaPtCld)) {
            visualizer->addPointCloud(lmksPtCld);
            visualizer->addPointCloud(pcaPtCld);
            visualizer->setPosition(0, 0);
            visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5);
            visualizer->setSize(pcaPtCld->width, pcaPtCld->height);
            visualizer->initCameraParameters();
        }

        std::cout << "Out PCARigidVisualizerFrontEnd::process\n";
    }

}