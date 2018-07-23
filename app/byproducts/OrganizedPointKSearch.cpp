#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <string>
#include <iostream>
#include <ostream>
#include <cuda_runtime_api.h>
#include <pcl/search/organized.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include "align/rigid_pipe.h"
#include "align/nonrigid_pipe.h"
#include "face/classify_pipe.h"
#include "feature/feature_detect_pipe.h"
#include "io/device.h"
#include "io/grabber.h"
#include "io/merger/device_input_merger.h"
#include "io/frontend.h"
#include "io/ply/meshio.h"
#include "io/align/align_frontend.h"
#include "cloud/cloud_pipe.h"
#include "face/cu_model_kernel.h"
#include "face/model_cudahelper.h"
#include "face/raw_model.h"

#include "mesh/mesh.h"
#include "mesh/color_projection_pipe.h"
#include "glog/logging.h"
#include "util/cudautil.h"
#include "util/eigen_file_io.h"

namespace {
    using namespace std;
    using namespace telef::io::align;
    using namespace telef::io;
    using namespace telef::cloud;
    using namespace telef::align;
    using namespace telef::face;
    using namespace telef::mesh;

    namespace fs = std::experimental::filesystem;

    namespace po = boost::program_options;
}
/** Loads an RGB image and a corresponding pointcloud. Make and write PLY face mesh out of it. */
int main(int ac, const char* const *av) {

    google::InitGoogleLogging(av[0]);

    float *d;
    CUDA_CHECK(cudaMalloc((void**)(&d), sizeof(float)));

    po::options_description desc("Captures RGB-D from camera. Generate and write face mesh as ply and obj");
    desc.add_options()
            ("help,H", "print this help message")
            ("model,M", po::value<std::string>(), "specify PCA model path")
            ("pcd,P", po::value<std::string>(), "specify PointCloud path")
            ("transform,T", po::value<std::string>(), "specify Rigid Transformation Matrix path")
            ("radius,R", po::value<double>(), "specify Radius to search")
            ("output,O", po::value<std::string>(), "specify output PLY file path");
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);


    if(vm.count("help") > 0) {
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("pcd") == 0) {
        std::cout << "Please specify 'path' to Pointcloud"  << std::endl;
        return 1;
    }

    if (vm.count("transform") == 0) {
        std::cout << "Please specify 'path' to Rigid Transformation Matrix"  << std::endl;
        return 1;
    }

    if (vm.count("output") == 0) {
        std::cout << "Please specify 'output'"  << std::endl;
        return 1;
    }

    std::string modelPath;
    std::string outputPath;
    std::string pcdPath("");
    std::string rigidPath("");
    double radius = 2e-2;

    modelPath = vm["model"].as<std::string>();
    outputPath = vm["output"].as<std::string>();
    pcdPath = vm["pcd"].as<std::string>();
    rigidPath = vm["transform"].as<std::string>();

    if (vm.count("radius") > 0) {
        radius = vm["radius"].as<double>();
    }

    std::shared_ptr<MorphableFaceModel> model;
    model = std::make_shared<MorphableFaceModel>(fs::path(modelPath.c_str()));

    auto transform = telef::util::readCSVtoEigan(rigidPath);

    double *avgShapeCoeff = new double[model->getShapeRank()]{0,};
    double *avgExprCoeff = new double[model->getExpressionRank()]{0,};
    auto mesh = model->genMesh(avgShapeCoeff, model->getShapeRank(), avgExprCoeff, model->getExpressionRank());

    //Transform Mesh
    mesh.applyTransform(*transform);
    std::cout << "\n Transformtion Matrix: \n" << *transform << std::endl;

    // Load cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudIn = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
    pcl::PCDReader pcdReader;

    unsigned int searchIdx;

    if (pcdReader.read (pcdPath, *cloudIn) == -1)
    {
        std::cout <<"Couldn't read input cloud" << std::endl;
        return -1;
    }

//    std::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA>> organizedNeighborSearch =
//            std::make_shared<pcl::search::OrganizedNeighbor<pcl::PointXYZRGBA>>();
//    organizedNeighborSearch->setInputCloud(cloudIn);


    C_ScanPointCloud scan;
    float fx = 571.401;
    float fy = 571.401;
    std::vector<int> scanLmkIdx;
//    std::vector<int> validLmks;
    Eigen::Matrix4f rigidTransform;
    int nMeshPoints = mesh.position.rows()/3;
    int nMeshSize = mesh.position.rows();

    //Host
    int *meshCorr_h = new int[nMeshSize];
    int *scanCorr_h = new int[nMeshSize];
    float *distance_h = new float[nMeshSize];
    int numCorr = 0;

    //Device
    int* meshCorr_d;
    int* scanCorr_d;
    float* distance_d;
    int* numCorr_d;

    float* mesh_d;

    pcl::PointCloud<PointT>::Ptr emptyLmks (new pcl::PointCloud<PointT>);


    CUDA_CHECK(cudaMalloc((void**)(&meshCorr_d), nMeshSize*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)(&scanCorr_d), nMeshSize*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)(&distance_d), nMeshSize*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)(&numCorr_d), sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)(&mesh_d), nMeshSize*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(mesh_d,mesh.position.data(), nMeshSize*sizeof(float), cudaMemcpyHostToDevice));

    loadScanToCUDADevice(&scan, cloudIn, fx, fy, scanLmkIdx, rigidTransform, emptyLmks);

    auto begin = chrono::high_resolution_clock::now();

    find_mesh_to_scan_corr(meshCorr_d, scanCorr_d, distance_d, numCorr_d, mesh_d, nMeshSize, scan, radius);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    auto dur = end - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    cout << "Closest Point Search Time (ms):" << ms << endl;

    CUDA_CHECK(cudaMemcpy(meshCorr_h, meshCorr_d, nMeshSize * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scanCorr_h, scanCorr_d, nMeshSize * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(distance_h, distance_d, nMeshSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&numCorr, numCorr_d, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(meshCorr_d));
    CUDA_CHECK(cudaFree(scanCorr_d));
    CUDA_CHECK(cudaFree(distance_d));
    CUDA_CHECK(cudaFree(mesh_d));

    freeScanCUDA(scan);

    pcl::PointCloud<PointT>::Ptr closestScanPoints (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr closestModelPoints (new pcl::PointCloud<PointT>);
    int nCorrPoints = 0;

    cout << "NumCorr:" << numCorr << endl;
    for (int idx = 0; idx < numCorr; idx++)
    {
            closestScanPoints->push_back(cloudIn->at(scanCorr_h[idx]));
            pcl::PointXYZRGBA searchPoint;
            searchPoint.x = mesh.position(3*meshCorr_h[idx]);
            searchPoint.y = mesh.position(3*meshCorr_h[idx]+1);
            searchPoint.z = mesh.position(3*meshCorr_h[idx]+2);
            searchPoint.r = 0;
            searchPoint.g = 255;
            searchPoint.b = 0;
            searchPoint.a = 0;
            closestModelPoints->push_back(searchPoint);
    }

    pcl::PLYWriter plyWriter;
    plyWriter.write(outputPath, *closestScanPoints);
    plyWriter.write("modelCorr.ply", *closestModelPoints);
    plyWriter.write("cloudin.ply", *cloudIn);
    delete [] meshCorr_h;
    delete [] distance_h;

    return 0;
}
