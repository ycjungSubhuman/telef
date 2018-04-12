#pragma once

#include <experimental/filesystem>
#include <vector>
#include <exception>

#include <Eigen/Core>

#include "mesh/mesh.h"
#include "io/ply/ply.h"

using namespace telef::mesh;
using fs = std::experimental::filesystem;

namespace {
    Eigen::Matrix2Xf getPositionAndColor(PlyFile *vertexElem, int numElem) {
        for (int i=0; i<numElem; i++) {
            ply_
        }
    }
    Eigen::VectorXf getPosition(PlyFile *vertexElem, int numElem) {

    }
    Eigen::Matrix3Xf getFace(PlyFile *faceElem, int numElem) {

    }
}

namespace telef::io::ply {
    ColorMesh readMesh(fs::path f) {
        int nelems;
        char **elem_names;
        int file_type;
        float version;

        std::vector<char> filename(f.begin(), f.end());
        PlyFile* ply = ply_open_for_reading(&filename[0], &nelems, &elem_names, &file_type, &version);

        if(ply == NULL) {
            throw std::runtime_error("File Open Failed");
        }

        // Read vertex element
        int numVertexElems;
        bool isColorMesh = false;
        {
            int numVertexProps;
            ply_get_element_description(ply, const_cast<char *>("vertex"), &numVertexElems, &numVertexProps);
            PlyProperty *vertexPropList;
            // Vertex element sanity check && Detect color properties
            assert(numVertexProps >= 3);
            assert(std::strcmp("x", vertexElement->props[0]->name) == 0);
            assert(std::strcmp("y", vertexElement->props[1]->name) == 0);
            assert(std::strcmp("z", vertexElement->props[2]->name) == 0);
            if (numVertexProps != 3) {
                assert(numVertexProps == 6);
                assert(std::strcmp("red", vertexElement->props[3]->name) == 0);
                assert(std::strcmp("green", vertexElement->props[4]->name) == 0);
                assert(std::strcmp("blue", vertexElement->props[5]->name) == 0);
                isColorMesh = true;
            }
        }

        // Read face element
        int numFaceElems;
        {
            int numFaceProps;
            PlyProperty *facePropList;
            ply_get_element_description(ply, const_cast<char *>("face"), &numFaceElems, &numFaceProps);
            // Face element sanity check
            assert(faceElement->props[0]->is_list == PLY_LIST);
        }

        ColorMesh mesh;

        // Read vertex element into Eigen matrix
        if(isColorMesh) {
            auto posCol = getPositionAndColor(ply, numVertexElems);
            mesh.position.swap(posCol.col(0));
            mesh.color.swap(posCol.col(1));
        }
        else {
            auto pos = getPosition(ply, numVertexElems);
            mesh.position.swap(pos.col(0));
        }


        auto face = getFace(ply, numFaceElems);

        ply_close(ply);
        return ColorMesh();
    }
}

