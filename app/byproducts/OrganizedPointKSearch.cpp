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

/**
 *   -name1
 *   path1
 *   path2
 *   ...
 *
 *   -name2
 *   path1
 *   path2
 *   ...
 *
 */
std::vector<std::pair<std::string, fs::path>> readGroups(fs::path p) {
    std::ifstream file(p);

    std::vector<std::pair<std::string, fs::path>> result;

    while(!file.eof()) {
        std::string word;
        file >> word;
        if (*word.begin() == '-') // name of group
        {
            std::string p;
            file >> p;
            result.push_back(std::make_pair(word, p));
        }
    }

    file.close();
    return result;
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
    double radius = 1e-2;

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

    std::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA>> organizedNeighborSearch =
            std::make_shared<pcl::search::OrganizedNeighbor<pcl::PointXYZRGBA>>();
    organizedNeighborSearch->setInputCloud(cloudIn);
    
    pcl::PointCloud<PointT>::Ptr closestScanPoints (new pcl::PointCloud<PointT>);

    auto begin = chrono::high_resolution_clock::now();
    int K = 16;
    for (int idx = 0; idx < mesh.position.rows()/3; idx++)
    {
        // define a random search point

        // generate point
        pcl::PointXYZRGBA searchPoint;
        searchPoint.x = mesh.position(3*idx);
        searchPoint.y = mesh.position(3*idx+1);
        searchPoint.z = mesh.position(3*idx+2);
        searchPoint.r = 0;
        searchPoint.g = 255;
        searchPoint.b = 0;
        searchPoint.a = 0;

        std::vector<int> k_indices;
        std::vector<float> k_sqr_distances;

        // organized nearest neighbor search
//        organizedNeighborSearch->nearestKSearch(searchPoint, (int)K, k_indices, k_sqr_distances);
        organizedNeighborSearch->radiusSearch(searchPoint, radius, k_indices, k_sqr_distances, (int)K);

        if (k_indices.size() == 0) {
            std::cout << "Failed to find closes point, " << idx << std::endl;
            continue;
        }
        int closest = 0;
        float dist = k_sqr_distances[closest];
        for (int i = 1; i < k_sqr_distances.size(); i++){
            if (k_sqr_distances[i] < dist) {
                dist = k_sqr_distances[i];
                closest = i;
            }
        }

        closestScanPoints->push_back(cloudIn->at(k_indices[closest]));
    }


    auto end = chrono::high_resolution_clock::now();
    auto dur = end - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    cout << "Closest Point Search Time (ms):" << ms << endl;

    pcl::PLYWriter plyWriter;
    plyWriter.write(outputPath, *closestScanPoints);
    plyWriter.write("cloudin.ply", *cloudIn);

    return 0;
}
