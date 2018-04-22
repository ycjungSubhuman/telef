#include "mesh/colormapping.h"

namespace {
    Eigen::Vector2i convertXyzToUv(const Eigen::Vector3f &xyz, float fx, float fy, float cx, float cy) {
        Eigen::Vector2i uv;
        uv(0) = static_cast<int>(std::round(xyz(0) * fx / xyz(2) + cx));
        uv(1) = static_cast<int>(std::round(xyz(1) * fy / xyz(2) + cy));
        return uv;
    }
}

namespace telef::mesh {
    
    void projectColor(ImagePtrT image, ColorMesh &mesh, float fx, float fy) {
        /* mesh to image space coordinate transform */
        float cx = (static_cast<float>(image->getWidth()) - 1.f) / 2.f;
        float cy = (static_cast<float>(image->getHeight()) - 1.f) / 2.f;
        const uint8_t *rgb_buffer = (const uint8_t*)image->getData();

        for (int i=0; i<mesh.position.rows(); i+=3) {
            float x = mesh.position(i);
            float y = mesh.position(i+1);
            float z = mesh.position(i+2);

            Eigen::Vector3f xyz;
            xyz << x,y,z;

            Eigen::Vector2i uv = convertXyzToUv(xyz, fx, fy, cx, cy);
            int pixel_idx = 3*(image->getWidth()*uv(1) + uv(0));

            uint8_t r = rgb_buffer[pixel_idx];
            uint8_t g = rgb_buffer[pixel_idx+1];
            uint8_t b = rgb_buffer[pixel_idx+2];
            mesh.color(i) = r;
            mesh.color(i+1) = g;
            mesh.color(i+2) = b;
        }
    }
}