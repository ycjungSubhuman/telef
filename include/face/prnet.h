#pragma once

#include <tensorflow/core/public/session.h>
#include <experimental/filesystem>
#include <Eigen/Core>
#include <vector>

// Wrapper for PRNet Face Position Map Regressor
namespace {
    namespace tf = tensorflow;
    namespace fs = std::experimental::filesystem;
}

namespace telef::face {
    class PRNetLandmarkDetector {
    public:
        PRNetLandmarkDetector(fs::path graphPath, fs::path checkpointPath);

        /**
         * Run the input image through this network
         *
         * image MUST have 256x256 size, and 3 channels for RGB
         *
         * @param image         rgbrgb...
         *
         * @return 3x68 landmark positions. Each column is (UV coordinate(dim2), Depth(dim1))
         */
        Eigen::MatrixXf Run(const float *image);
    private:
        static const std::vector<std::vector<int>> landmarkUVCoords;

        tf::Session *session;
    };
}