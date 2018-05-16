#pragma once

#include <Eigen/Core>
#include <math.h>

#include "mesh/mesh.h"
#include "type.h"

namespace {
    using namespace telef::types;
}

namespace telef::mesh {

    /**
     * Projects color from 2D RGB image onto aligned Colormesh
     *
     * image : RGB image
     * [out] mesh : scan-aligned mesh
     * fx : Horizontal focal length of the device in the model's coordinate
     * fy : Vertical focal length of the device
     */
    void projectColor(ImagePtrT image, ColorMesh &mesh, float fx, float fy);
}