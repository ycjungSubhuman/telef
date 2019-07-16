#pragma once

#include <Eigen/Dense>
#include "mesh/mesh.h"

namespace telef::util
{
    template<typename T>
    void colormesh2raw(
        const ColorMesh &colormesh,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &V,
        Eigen::MatrixXi &F)
    {
        size_t numvert = colormesh.position.size() / 3;
        V.resize(numvert, 3);
        for(size_t i=0; i<numvert; i++)
        {
            for(int k=0; k<3; k++)
            {
                V(i, k) = colormesh.position(3*i + k);
            }
        }

        size_t numtriangle = colormesh.triangles.size();
        F.resize(numtriangle, 3);
        for(size_t i=0; i<numtriangle; i++)
        {
            for(int k=0; k<3; k++)
            {
                F(i, k) = colormesh.triangles.at(i).at(k);
            }
        }
    }
}