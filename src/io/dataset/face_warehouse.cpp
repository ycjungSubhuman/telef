#include <fstream>
#include <string>

#include "io/dataset/face_warehouse.h"
#include "io/obj.h"
#include "util/require.h"

namespace telef::io::dataset {
namespace {
using namespace telef::mesh;
namespace fs = std::experimental::filesystem;

// blendshape binary file root path
fs::path bs_root(int testerIndex) {
  return fs::path("Tester_" + std::to_string(testerIndex + 1)) /
      fs::path("Blendshape");
}

// relative blendshape file path per identity
fs::path bs_filename(int testerIndex) {
  return bs_root(testerIndex) / fs::path("shape.bs");
}

// relative neutral face model file path per identity
fs::path neutral_filename(int testerIndex) {
  return bs_root(testerIndex) / fs::path("shape_0.obj");
}

// convert quad faces to triangle faces
Eigen::MatrixXi
quad_to_triangle(const Eigen::MatrixXf &V, const Eigen::MatrixXi &quads) {
  Eigen::MatrixXi result(2 * quads.rows(), 3);
  assert(4 == quads.cols());

  for (Eigen::Index faceIndex = 0; faceIndex < quads.rows(); faceIndex++) {
    Eigen::RowVectorXf A = V.row(quads(faceIndex, 0));
    Eigen::RowVectorXf B = V.row(quads(faceIndex, 1));
    Eigen::RowVectorXf C = V.row(quads(faceIndex, 2));
    Eigen::RowVectorXf D = V.row(quads(faceIndex, 3));

    // Choose ac
    result(2 * faceIndex, 0) = quads(faceIndex, 0);
    result(2 * faceIndex, 1) = quads(faceIndex, 1);
    result(2 * faceIndex, 2) = quads(faceIndex, 2);
    result(2 * faceIndex + 1, 0) = quads(faceIndex, 0);
    result(2 * faceIndex + 1, 1) = quads(faceIndex, 2);
    result(2 * faceIndex + 1, 2) = quads(faceIndex, 3);
  }

  return result;
}

// get triangle faces (array of vertex indices) from FaceWarehouse root path
Eigen::VectorXi get_tri_face(const fs::path &fw_root) {
  auto [tempVert, tempFace] = readObj(fw_root / neutral_filename(0));
  TELEF_REQUIRE(0 != tempFace.size());
  Eigen::MatrixXi tri = (4 == tempFace.cols())
      ? quad_to_triangle(tempVert, tempFace).transpose()
      : tempFace.transpose();

  Eigen::VectorXi face = Eigen::Map<Eigen::VectorXi>(tri.data(), tri.size());
  return face;
}

void normalize_position(
    Eigen::VectorXf ref,
    std::vector<std::vector<Eigen::VectorXf>> &positions,
    std::vector<int> lmkInds)
{

  Eigen::MatrixXf lmk_pts(lmkInds.size(), 3);
  for (int i = 0; i < lmkInds.size(); i++) {
    lmk_pts(i,0) = ref(3*lmkInds[i]+0);
    lmk_pts(i,1) = ref(3*lmkInds[i]+1);
    lmk_pts(i,2) = ref(3*lmkInds[i]+2);
  }

  for(size_t i=0; i<positions.size(); i++)
    {
      for(size_t j=0; j<positions[i].size(); j++)
        {
          Eigen::VectorXf v = positions[i][j];
          Eigen::Matrix3Xf mesh_pts_t =
            Eigen::Map<Eigen::Matrix3Xf>(v.data(), 3, v.size() / 3);
          Eigen::MatrixXf mesh_lmk_pts(lmkInds.size(), 3);
          for (int k = 0; k < lmkInds.size(); k++) {
            mesh_lmk_pts.row(k) = mesh_pts_t.col(lmkInds[k]);
          }
          Eigen::MatrixXf trans =
            Eigen::umeyama(mesh_lmk_pts.transpose(), lmk_pts.transpose(), false);
          Eigen::Matrix3Xf aligned =
            (trans*mesh_pts_t.colwise().homogeneous()).colwise().hnormalized();
          Eigen::VectorXf pos = 
            Eigen::Map<Eigen::VectorXf>(aligned.data(), aligned.size());
          positions[i][j] = pos;
        }
    }
}

std::vector<std::vector<int>> tri2vecvec(const Eigen::VectorXi &f) {
  std::vector<std::vector<int>> result(f.size() / 3);
  for (size_t i = 0; i < result.size(); i++) {
    result[i].resize(3);
    for (int j = 0; j < 3; j++) {
      result[i][j] = f(3 * i + j);
    }
  }
  return result;
}
} // namespace

FaceWarehouse::FaceWarehouse(fs::path root, std::optional<std::vector<int>> lmkInds)
    : m_root(std::move(root)), m_faces(get_tri_face(m_root)), m_idCount(150),
      m_bsCount(45) {
  for (int idIndex = 0; idIndex < m_idCount; idIndex++) {
    const auto bsPath = m_root / bs_filename(idIndex);
    TELEF_REQUIRE(fs::exists(bsPath));
    std::ifstream bsFile(bsPath.string(), std::ios::binary);

    // check sanity
    int32_t bsCount, vCount, fCount;
    bsFile.read(reinterpret_cast<char *>(&bsCount), sizeof(int32_t));
    bsCount-=1;
    TELEF_REQUIRE(45 == bsCount);
    bsFile.read(reinterpret_cast<char *>(&vCount), sizeof(int32_t));
    bsFile.read(reinterpret_cast<char *>(&fCount), sizeof(int32_t));

    std::vector<Eigen::VectorXf> bs_positions;
    // load vertex position
    for (int bsIndex = 0; bsIndex < m_bsCount+1; bsIndex++) {
      Eigen::VectorXf vertices(3 * vCount);
      bsFile.read(
          reinterpret_cast<char *>(vertices.data()),
          3 * vCount * sizeof(float));

      bs_positions.emplace_back(std::move(vertices));
    }

    m_positions.emplace_back(std::move(bs_positions));
  }

  // Calculate the mean expressions
  for (size_t i = 0; i < m_bsCount+1; i++) {
    Eigen::VectorXf bsAcc = m_positions[0][i];
    TELEF_REQUIRE(0 != bsAcc.size());
    for (size_t j = 0; j < m_idCount; j++) {
      bsAcc += m_positions[j][i];
    }
    const Eigen::VectorXf meanBs(bsAcc / static_cast<float>(m_idCount));

    if (0 == i)
      {
        m_ref = meanBs;
      }
    else
      {
        m_mean_exps.emplace_back(std::move(meanBs));
      }
  }

  if(lmkInds.has_value())
    {
      normalize_position(m_positions[0][0], m_positions, *lmkInds);
    }
}

ColorMesh FaceWarehouse::GetMesh(int idIndex, int bsIndex) const {
  ColorMesh result;
  result.position = m_positions[idIndex][bsIndex];
  result.triangles = tri2vecvec(m_faces);
  return result;
}

ColorMesh FaceWarehouse::GetNeutral(int idIndex) const {
  return GetMesh(idIndex, 0);
}
ColorMesh FaceWarehouse::GetMeanExp(int bsIndex) const {
  ColorMesh result;
  result.position = m_mean_exps[bsIndex];
  result.triangles = tri2vecvec(m_faces);
  return result;
}

ColorMesh FaceWarehouse::GetRef() const
{
  ColorMesh result;
  result.position = m_ref;
  result.triangles = tri2vecvec(m_faces);
  return result;
}

int FaceWarehouse::GetVertexCount() const { return m_mean_exps[0].size() / 3; }
int FaceWarehouse::GetIdCount() const { return m_idCount; }
int FaceWarehouse::GetBsCount() const { return m_bsCount; }

std::vector<ColorMesh>
FaceWarehouseAllSampler::SampleId(const BilinearModel &bm) {
  const int size = bm.GetIdCount();
  std::vector<ColorMesh> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = bm.GetNeutral(i);
  }
  return result;
}

std::vector<ColorMesh>
FaceWarehouseAllSampler::SampleEx(const BilinearModel &bm) {
  const int size = bm.GetBsCount();
  std::vector<ColorMesh> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = bm.GetMeanExp(i);
  }
  return result;
}
} // namespace telef::io::dataset
