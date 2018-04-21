#pragma once

#include <iostream>
#include <experimental/filesystem>

#include <Eigen/Dense>
#include <pcl/common/eigen.h>

#include "align/rigid.h"
#include "io/pipe.h"
#include "mesh/mesh.h"


namespace telef::align {
    /**
     * Rigid alignment of PCA Template to FittingSuite data
     */
    //template <int ShapeRank>
    class PCARigidFittingPipe : public telef::io::Pipe<telef::feature::FittingSuite, PCARigidAlignmentSuite> {
    public:

        PCARigidFittingPipe(std::shared_ptr<telef::face::MorphableFaceModel<150>> model);
        //PCARigidFittingPipe();

    private:
        using MModelTptr = std::shared_ptr<telef::face::MorphableFaceModel<150>>;
        using BaseT = telef::io::Pipe<telef::feature::FittingSuite, PCARigidAlignmentSuite>;
        using PtCldPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

        // Odd error =operator is deleted
        MModelTptr pca_model;
        telef::mesh::ColorMesh meanMesh;
        PtCldPtr initShape;
        // TODO: Keep last frame transformation matrix (Trans1 * Trans2)
        // or pointcloud in to optimize between frames?

        boost::shared_ptr<PCARigidAlignmentSuite> _processData(boost::shared_ptr<telef::feature::FittingSuite> in) override;
    };
}
