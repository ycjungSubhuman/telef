#pragma once

#include <boost/shared_ptr.hpp>
#include <memory>
#include <Eigen/Core>

#include "io/pipe.h"
#include "align/nonrigid_pipe.h"
#include "face/model.h"
#include "mesh/mesh.h"
#include "type.h"


namespace telef::mesh {

    using ProjectionSuite = struct ProjectionSuite {
        ColorMesh fitResult;
        std::shared_ptr<telef::face::MorphableFaceModel> pca_model;
        telef::types::ImagePtrT image;
        float fx;
        float fy;
    };

    class Fitting2ProjectionPipe : public telef::io::Pipe<telef::align::PCANonRigidFittingResult, ProjectionSuite> {
    private:
        boost::shared_ptr<ProjectionSuite> _processData(boost::shared_ptr<telef::align::PCANonRigidFittingResult> in) override;
    };

    class ColorProjectionPipe : public telef::io::Pipe<ProjectionSuite, telef::mesh::ColorMesh> {
    private:
        boost::shared_ptr<telef::mesh::ColorMesh> _processData(boost::shared_ptr<ProjectionSuite> in) override;
    };
}