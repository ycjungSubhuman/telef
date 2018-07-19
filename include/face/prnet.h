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

    enum PRNetFeatures {
        CHIN,
        L_BROW,
        R_BROW,
        NOSE,
        L_EYE,
        R_EYE,
        MOUTH
    };

    static const std::map<PRNetFeatures, std::vector<int>> PRNetLandmarkFeatures =
            {{CHIN, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}},
             {L_BROW, {17,18,19,20,21}},
             {R_BROW, {22,23,24,25,26}},
             {NOSE, {27,28,29,30,31,32,33,34,35}},
             {L_EYE, {36,37,38,39,40,41}},
             {R_EYE, {42,43,44,45,46,47}},
             {MOUTH, {48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67}}};

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