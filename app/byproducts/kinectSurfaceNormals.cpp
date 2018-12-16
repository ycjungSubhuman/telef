#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/conversions.h>
#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <string>
#include <ostream>

#include "glog/logging.h"
#include "util/po_util.h"

namespace {
    using namespace std;
    using namespace telef::util;
    namespace fs = std::experimental::filesystem;

    namespace po = boost::program_options;
}


void
saveCloud (const string &filename, const pcl::PCLPointCloud2 &output,
           const Eigen::Vector4f &translation, const Eigen::Quaternionf &orientation)
{
    pcl::PCDWriter w;
    w.writeBinaryCompressed (filename, output, translation, orientation);
}

int main(int ac, const char* const *av) {

    google::InitGoogleLogging(av[0]);

    po::options_description desc("Captures RGB-D from camera. Generate and write face mesh as ply and obj");
    desc.add_options()
            ("help,H", "print this help message")
            ("depthscan,D", po::value<std::string>(), "specify path to depth scan");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    require(vm, "depthscan");

    if(vm.count("help") > 0) {
        std::cout << desc << std::endl;
        return 1;
    }

    std::string scanPath = vm["depthscan"].as<std::string>();
    // load point cloud
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    // Load the first file
    Eigen::Vector4f translation;
    Eigen::Quaternionf rotation;

    pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
    pcl::io::loadPCDFile (scanPath, *cloud, translation, rotation);

    // Convert data to PointCloud<T>
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2 (*cloud, *xyz);

    // estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(xyz);
    ne.compute(*normals);


    // Convert data back
    pcl::PCLPointCloud2 output_normals;
    pcl::PCLPointCloud2 output;
    pcl::toPCLPointCloud2 (*normals, output_normals);
    pcl::concatenateFields (*cloud, output_normals, output);

    pcl::PCLImage image;
    pcl::PointCloud<pcl::PointNormal> outCloud;
    pcl::fromPCLPointCloud2 (output, outCloud);
    pcl::io::PointCloudImageExtractorFromNormalField<pcl::PointNormal> pcie;
    pcie.setPaintNaNsWithBlack (true);
    bool extracted = pcie.extract(outCloud, image);

    pcl::io::savePNGFile ("normals.png", image);


//
//
//    saveCloud ("output", output, translation, rotation);

//    // visualize normals
//    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
//    viewer.setBackgroundColor (0.0, 0.0, 0.0);
//    viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(xyz, normals);
//    while (!viewer.wasStopped ())
//    {
//        viewer.spinOnce ();
//    }
    return 0;
}