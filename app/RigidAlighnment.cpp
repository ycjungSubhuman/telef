//
// 3D Features & 3D Model Alignment (Rigid Registration)
// 3D Scan can be aligned using the same rotation and translation
//

#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include "pcl/common/geometry.h"
#include "pcl/common/transformation_from_correspondences.h"
#include "pcl/common/transforms.h"
#include "pcl/common/distances.h"
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/search/kdtree.h>

using namespace std;
using namespace cv;
//using namespace telef::feature;

bool fileExists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    //viewer->addCoordinateSystem (1.0, "global");
    viewer->initCameraParameters ();
    return (viewer);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr matToPointXYZ(cv::Mat cvPtCld)
{
    /*
    *  Function: Get from a Mat to pcl pointcloud datatype
    *  In: cv::Mat
    *  Out: pcl::PointCloud
    */

    //char pr=100, pg=100, pb=100;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptCld_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    //(new pcl::pointcloud<pcl::pointXYZ>);

    for(int i=0;i<cvPtCld.rows;i++)
    {

        pcl::PointXYZ point;
        point.x = cvPtCld.at<float>(i,0);
        point.y = cvPtCld.at<float>(i,1);
        point.z = cvPtCld.at<float>(i,2);

        //std::cout<<point.x<<", "<<point.y<<", "<<point.z<<endl;
        // when color needs to be added:
        //uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
        //point.rgb = *reinterpret_cast<float*>(&rgb);

        ptCld_ptr->points.push_back(point);


    }
    ptCld_ptr->width = (int)ptCld_ptr->points.size();
    ptCld_ptr->height = 1;

    return ptCld_ptr;
}

pcl::PolygonMesh::Ptr load_template(string fname) {
    if (! fileExists(fname)) {
        cout << "File doesn't exist " << fname << endl;
    }
    else {
        cout << "Loading " << fname << "...\n";
    }


    //pcl::TextureMesh mesh;
    //pcl::io::loadPolygonFileOBJ("3DScan_test3a.obj", mesh);

    // NOTE: The initial fitting should be of an average face!!
    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
    pcl::io::loadOBJFile(fname, *mesh);
    cout << "Loaded " << fname << "\n";

    return mesh;
}

boost::shared_ptr<cv::Mat> loadCvMatFromCSV(string fname) {
    if (! fileExists(fname)) {
        cout << "File doesn't exist " << fname << endl;
    }
    else {
        cout << "Loading " << fname << "...\n";
    }

    CvMLData mlData;
    mlData.read_csv(fname.c_str());

    //Seems Mat takes ownership of this pointer, if I delete it then it Seg Faults
    const CvMat* tmp = mlData.get_values();
    boost::shared_ptr<cv::Mat> cvMat(new cv::Mat(tmp, true));

    return cvMat;
}


int main(int argc, char** argv) {

    if (argc != 5) {
        cout << "Usage: <facemodel.obj> <3DLandmarks.csv> <lmk_face_idx.csv> <lmk_b_coords.csv>";
        return -1;
    }

    string model_fname = argv[1];
    string lmk_fname = argv[2];
    string lmk_faces_fname = argv[3];
    string lmk_b_coords_fname = argv[4];

    // Load Mean Face model
    pcl::PolygonMesh::Ptr mesh = load_template(model_fname);

    boost::shared_ptr<cv::Mat> lmks = loadCvMatFromCSV(lmk_fname);
    cout << "Loaded Landmarks: " /*<< *lmks */<< "\n";

    boost::shared_ptr<cv::Mat> lmk_faces_idx = loadCvMatFromCSV(lmk_faces_fname);
    cout << "Loaded lmk_faces_idx: " << lmk_faces_idx->rows << "\n";

    boost::shared_ptr<cv::Mat> lmk_b_coords = loadCvMatFromCSV(lmk_b_coords_fname);
    cout << "Loaded lmk_b_coords: " /*<< *lmk_b_coords */<< "\n";


    // Convert cv::Mat into PointCloud<PointT>
    pcl::PointCloud<pcl::PointXYZ>::Ptr lmksPtCld = matToPointXYZ(*lmks);


    // Convert PointCloud2 into PointCloud<PointT>
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh->cloud,*model_ptr);


    //Get only faces of model
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_lmk_points(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::CorrespondencesPtr lmk_corres (new pcl::Correspondences());
    lmk_corres->resize(lmk_faces_idx->rows);


    pcl::search::Search<pcl::PointXYZ>* kdtree = new pcl::search::KdTree<pcl::PointXYZ> ();
    unsigned int no_of_neighbors = 1;

    for (int idx = 0; idx < lmk_faces_idx->rows; idx++) {
        // current landmark index from Captured landmakrs
        auto face_idx = lmk_faces_idx->at<float>(idx);

        /*
         * Here we will interpolate the model's landmark positions
         * Using the barycentric coordinates and face indices
         * provided by the FLAME demonstration code.
         *
         * NOTE: Bary coods sum to 1
         */
        // 3 points per polygon face (Triangle)
        auto m_pol_verts = mesh->polygons[face_idx].vertices;
        pcl::PointXYZ point1 = model_ptr->points[m_pol_verts[0]];
        pcl::PointXYZ point2 = model_ptr->points[m_pol_verts[1]];
        pcl::PointXYZ point3 = model_ptr->points[m_pol_verts[2]];


        // Interpolated point holder
        pcl::PointXYZ mdl_lmk_point;

        // Interpolation of cartisian cords done using bary coods
        mdl_lmk_point.x = (point1.x * lmk_b_coords->at<float>(idx,0))
                       + (point2.x * lmk_b_coords->at<float>(idx,1))
                       + (point3.x * lmk_b_coords->at<float>(idx,2));
        mdl_lmk_point.y = (point1.y * lmk_b_coords->at<float>(idx,0))
                       + (point2.y * lmk_b_coords->at<float>(idx,1))
                       + (point3.y * lmk_b_coords->at<float>(idx,2));
        mdl_lmk_point.z = (point1.z * lmk_b_coords->at<float>(idx,0))
                       + (point2.z * lmk_b_coords->at<float>(idx,1))
                       + (point3.z * lmk_b_coords->at<float>(idx,2));

        //Add Interpolated cartisian point to pointcloud
        model_lmk_points->points.push_back(mdl_lmk_point);

        // Add correspondances between inderpolated lmk@indk and captured lmk@idx
        pcl::Correspondence corr;


        // Euclidean distance between lmks
        auto distance = pcl::geometry::distance(mdl_lmk_point, lmksPtCld->points[idx]);

        corr.index_query = idx;

        // Captured lmk
        corr.index_match = idx;
        corr.distance = distance;
        (*lmk_corres)[idx] = corr;
    }
    // Finalize model landmark pointcloud
    model_lmk_points->width = (int)model_lmk_points->points.size();
    model_lmk_points->height = 1;


    // Use SVD to estimate Rigid Transformation between interpolated
    Eigen::Matrix4f transformation;
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    svd.estimateRigidTransformation(*model_lmk_points, *lmksPtCld, *lmk_corres, transformation);

    cout << "Transformtion Matrix: " << transformation << endl;

    // Use resultant Transformtion Matrix from SVD to Model to rigid transform Model -> captured Landmarks
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*model_ptr, *transformed_cloud, transformation);


    //---------------Visualize--------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //viewer->addPolygonMesh(mesh, "Face", 0);

    viewer->addPointCloud(model_ptr, "Face", 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(model_lmk_points, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ> (model_lmk_points, single_color, "model_Landmarks");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "model_Landmarks");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(lmksPtCld, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (lmksPtCld, single_color1, "Landmarks");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Landmarks");



    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(transformed_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ> (transformed_cloud, single_color2, "ICP Result");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ICP Result");

    //viewer->addCorrespondences<pcl::PointXYZ>(model_ptr, lmksPtCld, *lmk_corres, "Correspondences");


    viewer->spin();

    /*
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    */

    return 0;
}