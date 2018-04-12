#pragma once

#include <experimental/filesystem>
#include <vector>
#include <exception>

#include <Eigen/Core>

#include "mesh/mesh.h"
#include "io/ply/YaPly.h"

using namespace telef::mesh;
namespace fs = std::experimental::filesystem;

namespace {
}

namespace telef::io::ply {
    ColorMesh readMesh(fs::path f) {
        yaply::PlyFile plyFile {f.c_str()};
        auto vertexElem = plyFile["vertex"];
        auto faceElem = plyFile["face"];
        assert (vertexElem.nrElements > 0);
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;
        std::vector<float> red;
        std::vector<float> green;
        std::vector<float> blue;
        std::vector<std::vector<int>> vertex_index;
        bool xSucc = vertexElem.getScalarProperty("x", x);
        bool ySucc = vertexElem.getScalarProperty("y", y);
        bool zSucc = vertexElem.getScalarProperty("z", z);
        bool rSucc = vertexElem.getScalarProperty("red", red);
        bool gSucc = vertexElem.getScalarProperty("green", green);
        bool bSucc = vertexElem.getScalarProperty("blue", blue);
        auto vi = faceElem.getProperty<yaply::PLY_PROPERTY_LIST<int32_t, int32_t>>("vertex_indices");

        assert (xSucc && ySucc && zSucc);
        assert (vi != nullptr);

        std::vector<float> position(x.size()*3);
        std::vector<float> color(red.size()*3);
        for (unsigned long i = 0; i < x.size(); i++) {
            position[i*3+0] = x[i];
            position[i*3+1] = y[i];
            position[i*3+2] = z[i];
            if(rSucc && gSucc && bSucc) {
                color[i * 3 + 0] = red[i];
                color[i * 3 + 1] = green[i];
                color[i * 3 + 2] = blue[i];
            }
        }

        vertex_index = vi->data;

        ColorMesh colorMesh;
        colorMesh.position = Eigen::Map<Eigen::VectorXf> (position.data(), position.size());
        colorMesh.color = Eigen::Map<Eigen::VectorXf> (color.data(), color.size());
        colorMesh.triangles = std::move(vertex_index);

        return colorMesh;
    }
}

