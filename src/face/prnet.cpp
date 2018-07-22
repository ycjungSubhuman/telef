#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <experimental/filesystem>

#include "face/prnet.h"

namespace {
    namespace tf = tensorflow;
    namespace fs = std::experimental::filesystem;

    template<typename T>
    inline T *checkNull(T *ptr, const std::string &message) {
        if(ptr) {
            return ptr;
        }
        else {
            throw std::runtime_error(message);
        }
    }

    inline void checkTFStatus(tf::Status status, const std::string &message) {
        if(!status.ok()) {
            throw std::runtime_error(message + ": " + status.error_message());
        }
    }
}

namespace telef::face {
    PRNetLandmarkDetector::PRNetLandmarkDetector(fs::path graphPath, fs::path checkpointPath) {
        session = checkNull(tf::NewSession(tf::SessionOptions()), "TF Session could not be created");

        tf::MetaGraphDef graphDef;
        checkTFStatus(tf::ReadBinaryProto(tf::Env::Default(), graphPath.string(), &graphDef),
                      "Error reading graph from " + graphPath.string());
        checkTFStatus(session->Create(graphDef.graph_def()), "Could not create GraphDef");

        tf::Tensor checkpointPathTensor(tf::DT_STRING, tf::TensorShape());
        checkpointPathTensor.scalar<std::string>()() = checkpointPath;
        checkTFStatus(session->Run({{ "save/Const", checkpointPathTensor },},
                                   {},
                                   { "save/restore_all" },
                                   nullptr),
                      "Error loading checkpoint");
    }

    Eigen::MatrixXf PRNetLandmarkDetector::Run(const float *image) {
        checkNull(image, "image is NULL. Call SetInputImage First.");

        // Convert image pointer to tf::Tensor input
        tf::Tensor x(tf::DT_FLOAT, tf::TensorShape({1, 256, 256, 3}));
        auto dst = x.flat<float>().data();
        std::copy_n(image, 3*256*256, dst);
        std::vector<std::pair<std::string, tf::Tensor>> inputs = {
                { "Placeholder", x },
        };

        // Run the input through PRNet
        std::vector<tf::Tensor> outputs;
        checkTFStatus(session->Run(inputs, {"resfcn256/Conv2d_transpose_16/Sigmoid"}, {}, &outputs),
                      "Error on evaluating position map");

        const float *positionMap = outputs[0].flat<float>().data();

        // Get Landmark 3D Positions
        Eigen::MatrixXf result;
        result.resize(3, 68);
        for(int i=0; i<landmarkUVCoords.size(); i++) {
            auto &coord = landmarkUVCoords[i];
            auto u = coord[0];
            auto v = coord[1];

            std::copy_n(&positionMap[v*3*256 + 3*u], 3, &result.data()[3*i]);
        }
        for(int i=0; i<68; i++) {
            float z = result(2,i);
            result(2, i) = 1.0f - z;
        }

        return result*256.0f*1.1f;
    }

    const std::vector<std::vector<int>> PRNetLandmarkDetector::landmarkUVCoords {
            {15,  96},
            {22,  118},
            {26,  141},
            {32,  165},
            {45,  183},
            {67,  190},
            {91,  188},
            {112, 187},
            {128, 193},
            {143, 187},
            {164, 188},
            {188, 190},
            {210, 183},
            {223, 165},
            {229, 141},
            {233, 118},
            {240, 96},
            {58,  49},
            {71,  42},
            {85,  39},
            {97,  40},
            {106, 42},
            {149, 42},
            {158, 40},
            {170, 39},
            {184, 42},
            {197, 49},
            {128, 59},
            {128, 73},
            {128, 86},
            {128, 96},
            {117, 111},
            {122, 113},
            {128, 115},
            {133, 113},
            {138, 111},
            {78,  67},
            {86,  60},
            {95,  61},
            {102, 65},
            {96,  68},
            {87,  69},
            {153, 65},
            {160, 61},
            {169, 60},
            {177, 67},
            {168, 69},
            {159, 68},
            {108, 142},
            {116, 131},
            {124, 127},
            {128, 128},
            {131, 127},
            {139, 131},
            {146, 142},
            {137, 148},
            {132, 150},
            {128, 150},
            {123, 150},
            {118, 148},
            {110, 141},
            {122, 135},
            {128, 134},
            {133, 135},
            {145, 142},
            {132, 143},
            {128, 142},
            {123, 143},
    };
}