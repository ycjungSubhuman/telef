
#include "io/align/align_frontend.h"

#include "pcl/common/transforms.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>

#include "face/model.h"
#include "io/frontend.h"
#include "util/eigen_pcl.h"

using namespace telef::align;
using namespace telef::io;
using namespace telef::face;

namespace telef::io::align {

void PCARigidVisualizerFrontEnd::process(InputPtrT input) {
  auto lmksPtCld = input->fittingSuite->landmark3d;

  ColorMesh meanMesh = input->pca_model->genMesh(
      Eigen::VectorXf::Zero(input->pca_model->getShapeRank()),
      Eigen::VectorXf::Zero(input->pca_model->getExpressionRank()));
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcaPtCld =
      telef::util::convert(meanMesh.position);

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(
      *pcaPtCld, *transformed_cloud, input->transformation);

  //        pcl::PCLPointCloud2 mesh;
  //        pcl::toPCLPointCloud2(pcaPointCloud, mesh);
  //        pcl::PolygonMesh polyMesh;
  //        polyMesh.cloud = mesh;
  //        polyMesh.polygons[];
  //
  //        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA>
  //        single_color(lmksPtCld, 255, 0, 0);

  if (!visualizer) {
    visualizer = std::make_unique<vis::PCLVisualizer>();
    visualizer->setBackgroundColor(0, 0, 0);
  }
  visualizer->spinOnce();
  if (!visualizer->updatePointCloud(lmksPtCld, "Landmarks")) {
    visualizer->addPointCloud(lmksPtCld, "Landmarks");
    visualizer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Landmarks");
  }

  if (!visualizer->updatePointCloud(transformed_cloud, "Mesh")) {
    visualizer->addPointCloud(transformed_cloud, "Mesh");
    visualizer->setPosition(0, 0);
    visualizer->setSize(transformed_cloud->width, transformed_cloud->height);
    visualizer->initCameraParameters();
  }

  if (!visualizer->updatePointCloud(input->rawCloud, "PC")) {
    visualizer->addPointCloud(input->rawCloud, "PC");
    visualizer->setPosition(0, 0);
    visualizer->initCameraParameters();
  }
}

void ColorMeshPlyWriteFrontEnd::process(
    ColorMeshPlyWriteFrontEnd::InputPtrT input) {
  std::cout << "Saved" << std::endl;
  fs::path p{outputPath.c_str()};
  ply::writePlyMesh(p, *input);
}
} // namespace telef::io::align
