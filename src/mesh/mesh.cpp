#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mesh/mesh.h"

namespace telef::mesh {
    void ColorMesh::applyTransform(Eigen::Matrix4f transform)
    {
        Eigen::Map<Eigen::Matrix3Xf> v(position.data(), 3, position.size()/3);
        Eigen::Matrix3Xf result = (transform * v.colwise().homogeneous()).colwise().hnormalized();
        position = Eigen::Map<Eigen::VectorXf>{result.data(), result.size()};
    }
}