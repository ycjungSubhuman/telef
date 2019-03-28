#include <iomanip>

#include "io/bmp.h"
#include "io/ply/meshio.h"

namespace {
using namespace telef::mesh;
namespace fs = std::experimental::filesystem;
} // namespace

namespace telef::io::ply {
ColorMesh readPlyMesh(fs::path f) {
  yaply::PlyFile plyFile{f.c_str()};
  auto vertexElem = plyFile["vertex"];
  auto faceElem = plyFile["face"];
  assert(vertexElem.nrElements > 0);
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> u;
  std::vector<float> v;
  std::vector<uint8_t> red;
  std::vector<uint8_t> green;
  std::vector<uint8_t> blue;
  std::vector<std::vector<int>> vertex_index;
  bool xSucc = vertexElem.getScalarProperty("x", x);
  bool ySucc = vertexElem.getScalarProperty("y", y);
  bool zSucc = vertexElem.getScalarProperty("z", z);
  bool uSucc = vertexElem.getScalarProperty("u", u);
  bool vSucc = vertexElem.getScalarProperty("v", v);
  bool rSucc = vertexElem.getScalarProperty("red", red);
  bool gSucc = vertexElem.getScalarProperty("green", green);
  bool bSucc = vertexElem.getScalarProperty("blue", blue);
  auto vi = faceElem.getProperty<yaply::PLY_PROPERTY_LIST<int32_t, int32_t>>(
      "vertex_indices");

  assert(xSucc && ySucc && zSucc);
  assert(vi != nullptr);

  std::vector<float> position(x.size() * 3);
  std::vector<float> uv(x.size() * 2);
  std::vector<uint8_t> color(red.size() * 3);
  for (unsigned long i = 0; i < x.size(); i++) {
    position[i * 3 + 0] = x[i];
    position[i * 3 + 1] = y[i];
    position[i * 3 + 2] = z[i];
    if (rSucc && gSucc && bSucc) {
      color[i * 3 + 0] = red[i];
      color[i * 3 + 1] = green[i];
      color[i * 3 + 2] = blue[i];
    }
    if (uSucc && vSucc) {
      uv[i * 2 + 0] = u[i];
      uv[i * 2 + 1] = v[i];
    }
  }

  vertex_index = vi->data;

  ColorMesh colorMesh;
  colorMesh.position =
      Eigen::Map<Eigen::VectorXf>(position.data(), position.size());
  colorMesh.color = color;
  colorMesh.uv = Eigen::Map<Eigen::VectorXf>(uv.data(), uv.size());
  colorMesh.triangles = std::move(vertex_index);

  return colorMesh;
}

void writePlyMesh(fs::path f, ColorMesh &mesh) {
  yaply::PlyFile plyFile;

  assert(mesh.position.size() % 3 == 0);
  plyFile["vertex"].nrElements = static_cast<size_t>(mesh.position.size() / 3);
  plyFile["vertex"].setScalars("x,y,z", mesh.position.data());
  if (mesh.color.size() != 0) {
    plyFile["vertex"].setScalars("red,green,blue", mesh.color.data());
  }
  plyFile["face"].nrElements = static_cast<size_t>(mesh.triangles.size());
  plyFile["face"].setList("vertex_indices", mesh.triangles);
  plyFile.save(f.c_str(), false);
}

void writeObjMesh(fs::path f, ColorMesh &mesh) {
  auto stripped = f.parent_path() / f.stem();
  std::ofstream of(f.c_str());
  of << "mtllib " << f.stem().string() << ".mtl\n";
  of << "usemtl material0"
     << "\n";
  of << std::fixed;
  for (int i = 0; i < mesh.position.size() / 3; i++) {
    of << "v " << mesh.position[3 * i] << " " << mesh.position[3 * i + 1] << " "
       << mesh.position[3 * i + 2] << "\n";
  }

  for (int i = 0; i < mesh.position.size() / 3; i++) {
    of << "vt " << mesh.uv[2 * i] << " " << mesh.uv[2 * i + 1] << "\n";
  }

  for (int i = 0; i < mesh.triangles.size(); i++) {
    auto tri = mesh.triangles[i];
    of << "f " << (tri[0] + 1) << "/" << (tri[0] + 1) << " " << (tri[1] + 1)
       << "/" << (tri[1] + 1) << " " << (tri[2] + 1) << "/" << (tri[2] + 1)
       << "\n";
  }
  of.close();

  std::ofstream mtl(stripped.string() + ".mtl");
  mtl << "newmtl material0\n";
  mtl << "Ka 1.000000 1.000000 1.000000\n";
  mtl << "Kd 1.000000 1.000000 1.000000\n";
  mtl << "Ks 0.000000 0.000000 0.000000\n";
  mtl << "Tr 1.000000\n";
  mtl << "illum 1\n";
  mtl << "Ns 0.000000\n";
  mtl << "map_Kd " << f.stem().string() << ".jpg\n";
  mtl.close();

  if (mesh.image != nullptr) {
    saveBMPFile(stripped.string() + ".jpg", *mesh.image);
  }
}
} // namespace telef::io::ply
