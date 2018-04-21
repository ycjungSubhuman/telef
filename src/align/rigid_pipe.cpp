
#include <iostream>

#include <pcl/registration/transformation_estimation_svd.h>

#include "align/rigid_pipe.h"
#include "util/eigen_pcl.h"
#include "face/model.h"

using namespace std;
using namespace telef::feature;
using namespace telef::types;
using namespace telef::face;


namespace telef::align {
    PCARigidFittingPipe::PCARigidFittingPipe(MModelTptr model):
            BaseT(),
            pca_model(model)
    {
        //pca_model = std::make_shared<telef::face::MorphableFaceModel<150>>(fs::path("../pcamodels/example"));

        // Generate Mean Face Template
        meanMesh = pca_model->genMesh(Eigen::VectorXf::Zero(150));

        // Save initial point cloud for rigid fitting
        initShape = telef::util::convert(meanMesh.position);
    }

    boost::shared_ptr<PCARigidAlignmentSuite> PCARigidFittingPipe::_processData(boost::shared_ptr<FittingSuite> in) {
        std::vector<int> pca_lmks = pca_model->getLandmarks();
        auto in_lmks = in->landmark3d;

        std::vector<int> corr_tgt(in_lmks->points.size());
        std::iota(std::begin(corr_tgt), std::end(corr_tgt), 0); // Fill with range 0, ..., n.

        Eigen::Matrix4f transformation;
        pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZRGBA> svd;
        svd.estimateRigidTransformation(*initShape, pca_lmks, *in_lmks, corr_tgt, transformation);

        std::cout << "\n Transformtion Matrix: \n" << transformation << std::endl;

        auto alignment = boost::shared_ptr<PCARigidAlignmentSuite>(new PCARigidAlignmentSuite());
        alignment->fittingSuite = in;
        alignment->pca_model = pca_model;
        alignment->transformation = transformation;
        return alignment;
    }
}