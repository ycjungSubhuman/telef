#pragma once

#include <experimental/filesystem>

#include <Eigen/Dense>

#include <dlib/dnn.h>
#include <boost/asio.hpp>

#include "io/pipe.h"
#include "feature/face.h"

#include "face/prnet.h"


namespace {
    using namespace dlib;
}

namespace telef::feature {

    /**
     * Face Detection using Open CV haar cascade
     */
    class FaceDetectionPipe : public telef::io::Pipe<telef::io::DeviceInputSuite, FeatureDetectSuite> {
    protected:
        using BaseT = telef::io::Pipe<telef::io::DeviceInputSuite, Feature>;
        using InputPtrT = telef::io::DeviceInputSuite::Ptr;
    };

    /**
     * Face Detection using Open CV haar cascade
     */
//    class HaarFaceDetectionPipe : public FaceDetectionPipe {
//    private:
//        using BaseT = FaceDetectionPipe::BaseT;
//        using InputPtrT = FaceDetectionPipe::InputPtrT;
//
//        FaceDetectionSuite::Ptr _processData(InputPtrT in) override;
//
//    public:
//        HaarFaceDetectionPipe(const std::string &pretrained_model, const bool faceProfile);
//    };

    /**
     * Face Detection using Dlib CNN, realtime on Titain X GPU ~20ms per frame
     */
    class DlibFaceDetectionPipe : public FaceDetectionPipe {
    private:
        using BaseT = FaceDetectionPipe::BaseT;
        using InputPtrT = FaceDetectionPipe::InputPtrT;

        template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
        template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
        template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
        template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

        using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

        net_type net;

        //TODO: Keep old face location incase CNN fails???

        FeatureDetectSuite::Ptr _processData(InputPtrT in) override;

    public:
        DlibFaceDetectionPipe(const std::string &pretrained_model);
    };

    /**
     * Fake Face Feature Detection
    */
    class DummyFeatureDetectionPipe : public telef::io::Pipe<FeatureDetectSuite, FeatureDetectSuite> {
    private:
        using BaseT = telef::io::Pipe<FeatureDetectSuite, FeatureDetectSuite>;
        using InputPtrT = FeatureDetectSuite::Ptr;
        std::queue<Eigen::MatrixXf> frameLmks;

        FeatureDetectSuite::Ptr _processData(InputPtrT in) override;

    public:
        DummyFeatureDetectionPipe(fs::path recordPath);
    };

    /**
     * PRNet Face Feature Detection
    */
    class PRNetFeatureDetectionPipe : public telef::io::Pipe<FeatureDetectSuite, FeatureDetectSuite> {
    private:
        using BaseT = telef::io::Pipe<FeatureDetectSuite, FeatureDetectSuite>;
        using InputPtrT = FeatureDetectSuite::Ptr;

        telef::face::PRNetLandmarkDetector lmkDetector;
        int prnetIntputSize;
        Eigen::MatrixXf landmarks;

        FeatureDetectSuite::Ptr _processData(InputPtrT in) override;

        void calculateTransformation(cv::Mat& transform, const cv::Mat& image, const BoundingBox& bbox, const int dst_size);
        void warpImage(cv::Mat& warped, const cv::Mat& image, const cv::Mat& transform, const int dst_size);
        void restore(Eigen::MatrixXf& restored, const Eigen::MatrixXf& result, const cv::Mat& transformation);

    public:
        PRNetFeatureDetectionPipe(fs::path graphPath, fs::path checkpointPath);
    };

    /**
     * Face Feature Detection Client
    */
    class FeatureDetectionClientPipe : public telef::io::Pipe<FeatureDetectSuite, FeatureDetectSuite> {
    private:
        using BaseT = telef::io::Pipe<FeatureDetectSuite, FeatureDetectSuite>;
        using InputPtrT = FeatureDetectSuite::Ptr;
        // Local Unix Socket for IPC
        using SocketT = boost::asio::local::stream_protocol::socket;

        bool isConnected;
        std::string address;
        boost::asio::io_service &ioService;
        std::shared_ptr<SocketT> clientSocket;
        //int clientIntputSize;
        uint32_t msg_id;

        Eigen::MatrixXf landmarks;

        FeatureDetectSuite::Ptr _processData(InputPtrT in) override;

        // Connection
        bool connect();
        void disconnect();

//        // io
//        bool read_with_timeout(boost::asio::streambuf& buffer);
//        void set_result(std::optional<std::error_code>* a, std::error_code b);

    public:
        FeatureDetectionClientPipe(std::string address, boost::asio::io_service &service);
//        virtual ~FeatureDetectionClientPipe();
    };

}
