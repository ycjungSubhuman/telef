#pragma once

#include <experimental/filesystem>

#include <Eigen/Dense>
#include <pcl/common/eigen.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include "align/rigid.h"
#include "io/pipe.h"
#include "face/model.h"
#include "util/eigen_pcl.h"
#include "type.h"

using namespace telef::feature;
using namespace telef::types;
using namespace telef::face;

namespace telef::align {

    /**
     * Rigid alignment of PCA Template to FittingSuite data
     */
    //template <int ShapeRank>
    class PCARigidFittingPipe : public Pipe<FittingSuite, PCARigidAlignmentSuite> {
    public:
        using BaseT = Pipe<FittingSuite, PCARigidAlignmentSuite>;
        using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

//        PCARigidFittingPipe(MorphableFaceModel<150> model) {
        PCARigidFittingPipe() {
            pca_model = std::make_shared<MorphableFaceModel<150>>(fs::path("../pcamodels/example"));

            // Generate Mean Face Template
            meanMesh = pca_model->genMesh(Eigen::VectorXf::Zero(150));

            // Save initial point cloud for rigid fitting
            initShape = telef::util::convert(meanMesh.position);
        }

    private:
//        MorphableFaceModel<150> pca_model;
        // Odd error = operator is deleted
        std::shared_ptr<MorphableFaceModel<150>> pca_model;// = MorphableFaceModel<150>(fs::path("../pcamodels/example"));
        ColorMesh meanMesh;
        PtCldPtr initShape;
        // TODO: Keep last frame transformation matrix (Trans1 * Trans2) or pointcloud in to optimize between frames?


        virtual PCARigidAlignmentSuite::Ptr _processData(FittingSuite::Ptr in) {
            std::vector<int> pca_lmks = pca_model->getLandmarks();
            auto in_lmks = in->landmark3d;

            std::vector<int> corr_tgt(in_lmks->points.size());
            std::iota (std::begin(corr_tgt), std::end(corr_tgt), 0); // Fill with range 0, ..., n.

            Eigen::Matrix4f transformation;
            pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZRGBA> svd;
            svd.estimateRigidTransformation(*initShape, pca_lmks, *in_lmks, corr_tgt, transformation);

            cout << "\n Transformtion Matrix: \n" << transformation << endl;

            auto alignment = std::make_shared<PCARigidAlignmentSuite>();
            alignment->fittingSuite = in;
            alignment->pca_model = pca_model;
            alignment->transformation = transformation;

            return alignment;
        }
    };
}
