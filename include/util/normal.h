#pragma once
#include <vector>
#include "mesh/mesh.h"

namespace
{
using namespace telef::mesh;
}

namespace telef::util
{
inline std::vector<float> getVertexNormal(ColorMesh mesh) {
  std::vector<float> vertexNormal(
      static_cast<unsigned long>(mesh.position.size()), 0.0f);
  for (int i = 0; i < mesh.triangles.size(); i++) {
    const auto v1Ind = mesh.triangles[i][0];
    const auto v2Ind = mesh.triangles[i][1];
    const auto v3Ind = mesh.triangles[i][2];
    Eigen::Vector3f v1 = mesh.position.segment(3 * v1Ind, 3);
    Eigen::Vector3f v2 = mesh.position.segment(3 * v2Ind, 3);
    Eigen::Vector3f v3 = mesh.position.segment(3 * v3Ind, 3);
    Eigen::Vector3f unnormalizedNormal = (v2 - v1).cross(v3 - v1).normalized();
    vertexNormal[3 * v1Ind + 0] += unnormalizedNormal[0];
    vertexNormal[3 * v1Ind + 1] += unnormalizedNormal[1];
    vertexNormal[3 * v1Ind + 2] += unnormalizedNormal[2];

    vertexNormal[3 * v2Ind + 0] += unnormalizedNormal[0];
    vertexNormal[3 * v2Ind + 1] += unnormalizedNormal[1];
    vertexNormal[3 * v2Ind + 2] += unnormalizedNormal[2];

    vertexNormal[3 * v3Ind + 0] += unnormalizedNormal[0];
    vertexNormal[3 * v3Ind + 1] += unnormalizedNormal[1];
    vertexNormal[3 * v3Ind + 2] += unnormalizedNormal[2];
  }

  return vertexNormal;
}
}
