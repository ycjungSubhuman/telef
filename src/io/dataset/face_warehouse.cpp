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

const std::vector<int> fw_lmks = {
    1089,  1054,  1148,  11290, 3855, 3880, 504,  10573, 9150, 9078,
    6770,  9185,  6738,  6729,  8046, 8043, 2514, 702,   711,  10855,
    724,   10852, 7162,  7189,  2150, 9417, 9405, 9397,  8943, 8952,
    8972,  10488, 10487, 8922,  6620, 8923, 4359, 4097,  4087, 3935,
    10879, 10881, 9223,  9256,  2043, 6906, 9447, 7223,  3196, 3237,
    3254,  6072,  6142,  6124,  8811, 8865, 6119, 8815,  3233, 10298,
    164,   10334, 8802,  6171,  8835, 8918, 6267, 3376,
};

/*
void normalize_position(
    const std::vector<std::vector<Eigen::VectorXf>> &positions)
{
  for(size_t i=0; i<m_positions.size(); i++)
    {
      for(size_t j=0; j<m_positions[i].size(); j++)
        {
          Eigen::VectorXf input = m_meshes[i][j];

          Eigen::Matrix3f R;
          Eigen::Vector3f t;
          fext::fitting::Align3DMesh(
              R, t,
              *ref, *input, lmk3d, false);

          Eigen::Matrix3Xf p = input->GetPosition();
          Eigen::Matrix3Xf transp = (R*p).colwise() + t;
          auto transMesh =
            CMesh::Create(transp, ref->GetFaces(), 3);
          m_meshes[i][j] = transMesh;
        }
      fext::io::writeObj(std::to_string(i)+".obj", *m_meshes[i][0]);
    }
}
*/
} // namespace

FaceWarehouse::FaceWarehouse(fs::path root)
    : m_root(std::move(root)), m_faces(get_tri_face(m_root)), m_idCount(150),
      m_bsCount(46) {
  for (int idIndex = 0; idIndex < m_idCount; idIndex++) {
    const auto bsPath = m_root / bs_filename(idIndex);
    TELEF_REQUIRE(fs::exists(bsPath));
    std::ifstream bsFile(bsPath.string(), std::ios::binary);

    // check sanity
    int32_t bsCount, vCount, fCount;
    bsFile.read(reinterpret_cast<char *>(&bsCount), sizeof(int32_t));
    bsFile.read(reinterpret_cast<char *>(&vCount), sizeof(int32_t));
    bsFile.read(reinterpret_cast<char *>(&fCount), sizeof(int32_t));

    std::vector<Eigen::VectorXf> bs_positions;
    // load vertex position
    for (int bsIndex = 0; bsIndex < m_bsCount; bsIndex++) {
      Eigen::VectorXf vertices(3 * vCount);
      bsFile.read(
          reinterpret_cast<char *>(vertices.data()),
          3 * vCount * sizeof(float));
      bs_positions.emplace_back(std::move(vertices));
    }

    m_positions.emplace_back(std::move(bs_positions));
  }

  // Calculate the mean expressions
  for (size_t i = 0; i < m_bsCount; i++) {
    Eigen::VectorXf bsAcc = m_positions[0][i];
    for (size_t j = 0; j < m_idCount; j++) {
      bsAcc += m_positions[j][i];
    }
    const Eigen::VectorXf meanBs(bsAcc / static_cast<float>(m_idCount));
    m_mean_exps.emplace_back(std::move(meanBs));
  }

  // normalize_position(m_positions);
}

namespace {
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
