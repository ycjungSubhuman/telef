#include "mesh/colormapping.h"

namespace {
Eigen::Vector2i convertXyzToUv(const Eigen::Vector3f &xyz, float fx, float fy,
                               float cx, float cy) {
  Eigen::Vector2i uv;
  uv(0) = static_cast<int>(std::round(xyz(0) * fx / xyz(2) + cx));
  uv(1) = static_cast<int>(std::round(xyz(1) * fy / xyz(2) + cy));
  return uv;
}
} // namespace

namespace telef::mesh {

void projectColor(ImagePtrT image, ColorMesh &mesh, float fx, float fy) {
  /* mesh to image space coordinate transform */
  float cx = (static_cast<float>(image->getWidth()) - 1.f) / 2.f;
  float cy = (static_cast<float>(image->getHeight()) - 1.f) / 2.f;
  const uint8_t *rgb_buffer = (const uint8_t *)image->getData();

  mesh.color.resize(static_cast<unsigned long>(mesh.position.rows()));
  mesh.uv.resize(mesh.position.rows() / 3 * 2);
  for (int i = 0; i < mesh.position.rows() / 3; i++) {
    float x = mesh.position(3 * i);
    float y = mesh.position(3 * i + 1);
    float z = mesh.position(3 * i + 2);

    Eigen::Vector3f xyz;
    xyz << x, y, z;

    Eigen::Vector2i uv = convertXyzToUv(xyz, fx, fy, cx, cy);
    int pixel_idx = 0;
    if (uv(0) >= 0 && uv(0) < image->getWidth() && uv(1) >= 0 &&
        uv(1) < image->getHeight()) {
      pixel_idx = 3 * (image->getWidth() * uv(1) + uv(0));
    }

    uint8_t r = rgb_buffer[pixel_idx];
    uint8_t g = rgb_buffer[pixel_idx + 1];
    uint8_t b = rgb_buffer[pixel_idx + 2];
    mesh.color[3 * i] = r;
    mesh.color[3 * i + 1] = g;
    mesh.color[3 * i + 2] = b;
    mesh.uv[2 * i] = (static_cast<float>(uv(0)) / image->getWidth());
    mesh.uv[2 * i + 1] =
        1.0f - (static_cast<float>(uv(1)) / image->getHeight());
    mesh.image = image;
  }
}
} // namespace telef::mesh