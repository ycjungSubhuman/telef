

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/gui_widgets.h>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <boost/asio.hpp>

#include "util/eigen_file_io.h"
#include "util/pcl_cv.h"
#include "feature/feature_detect_pipe.h"


#include "messages/messages.pb.h"

using namespace std;
using namespace dlib;
using namespace telef::io;
using namespace boost::asio;


namespace telef::feature {
    /**
     * DlibFaceDetectionPipe
     * @param pretrained_model
     */
    DlibFaceDetectionPipe::DlibFaceDetectionPipe(const std::string &pretrained_model) {
        deserialize(pretrained_model) >> net;
    }

    FeatureDetectSuite::Ptr
    DlibFaceDetectionPipe::_processData(InputPtrT in) {
        matrix<rgb_pixel> img;

        // Convert PCL image to dlib image
        auto pclImage = in->rawImage;

        auto matImg = cv::Mat(pclImage->getHeight(), pclImage->getWidth(), CV_8UC3);
        pclImage->fillRGB(matImg.cols, matImg.rows, matImg.data, matImg.step);
        cv::cvtColor(matImg, matImg, CV_RGB2BGR);
        dlib::cv_image<bgr_pixel> cvImg(matImg);
        dlib::assign_image(img, cvImg);


        auto dets = net(img);
        dlib::rectangle bbox;
        double detection_confidence = 0;

        // Get best face detected
        // TODO: Handle face detection failure
        for (auto &&d : dets) {
            if (d.detection_confidence > detection_confidence) {
                detection_confidence = d.detection_confidence;
                bbox = d.rect;
            }
        }

        FeatureDetectSuite::Ptr result = boost::make_shared<FeatureDetectSuite>();

        result->deviceInput = in;
        result->feature = boost::make_shared<Feature>();
        result->feature->boundingBox.setBoundingBox(bbox);

        return result;
    }

    /**
     * Dummy Detector path based on PRNet (Python), Doesn't give compatible landmarks with our FW
     *
     * @param recordPath, Path given to fake device with captured landmarks
     */
    DummyFeatureDetectionPipe::DummyFeatureDetectionPipe(fs::path recordPath) : frameLmks() {
        std::cout << "Loading Fake Landmarks..." << std::endl;
        for(int i=1; ; i++) {
            fs::path framePath = recordPath/(fs::path(std::to_string(i)));

            auto exists = fs::exists(framePath.replace_extension(".3dlmk.txt"));

            if(!exists) {
                break;
            }

            auto lmks = telef::util::readCSVtoEigan(framePath);
            frameLmks.push(*lmks);
            std::cout << "Loaded landmarks " << std::to_string(i) << std::endl;
        }
    }

    FeatureDetectSuite::Ptr DummyFeatureDetectionPipe::_processData(InputPtrT in) {
        auto currentLmks = frameLmks.front();

        in->feature->points = currentLmks;
        frameLmks.pop();

        return in;
    }

    PRNetFeatureDetectionPipe::PRNetFeatureDetectionPipe(fs::path graphPath, fs::path checkpointPath)
            : lmkDetector(graphPath, checkpointPath), prnetIntputSize(256) { }

    void PRNetFeatureDetectionPipe::calculateTransformation(cv::Mat& transform,
                                                            const cv::Mat& image,
                                                            const BoundingBox& bbox,
                                                            const int dst_size){
        dlib::rectangle rect = bbox.getRect();
        long left = rect.left(), right = rect.right(), top = rect.top(), bottom = rect.bottom();
        float old_size = (right - left + bottom - top)/2.f;
        float center[2] = {right - rect.width() / 2.f, bottom - rect.height() / 2.f + old_size*0.14f};
        int size = int(old_size*1.58f);

        // Use std::vector<cv::Point> to tell estimateRigidTransform it is a set of points, not an image
        std::vector<cv::Point2f> src_pts;
        src_pts.emplace_back(center[0]-size/2, center[1]-size/2);
        src_pts.emplace_back(center[0] - size/2, center[1]+size/2);
        src_pts.emplace_back(center[0]+size/2, center[1]-size/2);

        std::vector<cv::Point2f> dst_pts;
        dst_pts.emplace_back(0,0);
        dst_pts.emplace_back(0, dst_size - 1);
        dst_pts.emplace_back(dst_size - 1, 0);

        // Compute 5-DOF similarity transform (translation, rotation, and uniform scaling)
        // basicially if FullAffine=True, then it will compute 6-DOF including sheer and we don't want sheer
        // The result is a 2x3 affine matrix
        transform = cv::estimateRigidTransform(src_pts, dst_pts, /*Full Affine*/false);
    }

    void PRNetFeatureDetectionPipe::warpImage(cv::Mat& warped,
                   const cv::Mat& image,
                   const cv::Mat& transform,
                   const int dst_size){

        // Convert CV_8UC3 to CV32FC3, which automatically normalizes intensities between 0 and 1
        cv::Mat normImage;
        image.convertTo(normImage, CV_32FC3);
        normImage /= 255.0f;

        // Transform is automatically inverted
        // Takes 2x3 Matrix
        cv::warpAffine(normImage, warped, transform, cv::Size(dst_size, dst_size) );
    }

    void PRNetFeatureDetectionPipe::restore(Eigen::MatrixXf& restored, const Eigen::MatrixXf& result, const cv::Mat& transform){
        cv::Mat resultMat;
        // Cast result to double to match cv::datatypes
        cv::eigen2cv(result, resultMat);

        // We are expecting only uniform scaling, so we use the x scaling term
        float scale = transform.at<float>(0,0);
        cv::Mat z;
        resultMat.row(2).copyTo(z);
        z /= scale;
        resultMat.row(2).setTo(cv::Scalar(1.0));
        cv::Mat verticies = transform.inv() * resultMat;
        z.copyTo(verticies.row(2));

        cv::cv2eigen(verticies, restored);
    }

    FeatureDetectSuite::Ptr PRNetFeatureDetectionPipe::_processData(InputPtrT in) {
        auto pclImage = in->deviceInput->rawImage;

        if (in->feature->boundingBox.width <= 0 || in->feature->boundingBox.height <= 0) {
            std::cout << "Face Detection Failed, returning previous PRNet landmarks...\n";
            in->feature->points = landmarks;
            return in;
        }

        auto matImg = telef::util::convert(pclImage);

        cv::Mat transform; // = cv::Mat::eye(3,3,CV_64F);
        calculateTransformation(transform, *matImg, in->feature->boundingBox, prnetIntputSize);

        cv::Mat warped;
        warpImage(warped, *matImg, transform, prnetIntputSize);

        // Detect Landmarks
        Eigen::MatrixXf result = lmkDetector.Run((float*)warped.data);


        // extend rigid transformation to use perspective transform for square matrix for inverting,
        // also convert to float from double returned from estimation for compatable matmul with out floating point data
        cv::Mat squareTransf = cv::Mat::eye(3,3,CV_32F);
        transform.convertTo(squareTransf.rowRange(0,2), CV_32F);

        restore(landmarks, result, squareTransf );

        in->feature->points = landmarks;
        return in;
    }


    FeatureDetectionClientPipe::FeatureDetectionClientPipe(string address_, boost::asio::io_service &service)
        : isConnected(false), address(address_), ioService(service), clientSocket(), msg_id(0) { }

//    FeatureDetectionClientPipe::~FeatureDetectionClientPipe(){
//        // Cleanly close connection
//        disconnect();
//    };

    FeatureDetectSuite::Ptr FeatureDetectionClientPipe::_processData(FeatureDetectionClientPipe::InputPtrT in) {

        if ( clientSocket == nullptr || !isConnected ) {
            if (connect()) {
                return in;
            }
        }

        // Convert PCL image to dlib image
        auto pclImage = in->deviceInput->rawImage;
        std::vector<unsigned char> imgBuffer(pclImage->getDataSize());


//        auto matImg = cv::Mat(pclImage->getHeight(), pclImage->getWidth(), CV_8UC3);
//        pclImage->fillRGB(matImg.cols, matImg.rows, matImg.data, matImg.step);
//        cv::cvtColor(matImg, matImg, CV_RGB2BGR);

        pclImage->fillRaw(imgBuffer.data());

        LmkReq reqMsg;
        auto hdr = reqMsg.mutable_hdr();
        hdr->set_id(msg_id++);
//        hdr->set_width(matImg.cols);
//        hdr->set_height(matImg.rows);
//        hdr->set_channels(matImg.channels());
        hdr->set_width(pclImage->getHeight());
        hdr->set_height(pclImage->getWidth());
        hdr->set_channels(3);


        auto imgData = reqMsg.mutable_data();
//        imgData->set_buffer(matImg.data, matImg.total());
        imgData->set_buffer(imgBuffer.data(), pclImage->getDataSize());

        try {
            // serialize
            boost::asio::streambuf sbuf;
            ostream serialized(&sbuf);
            reqMsg.SerializeToOstream(&serialized);

            uint32_t req_length = sbuf.size();

            req_length = htonl(req_length); // host -> net endianness!

            // Send header with request message size
            boost::asio::write(*clientSocket, boost::asio::buffer(&req_length, sizeof(std::uint32_t)));
            boost::asio::write(*clientSocket, sbuf);

            // read header for expected rsp size
            uint32_t rsp_length;
            boost::asio::read(*clientSocket, boost::asio::buffer(&rsp_length, sizeof(std::uint32_t)));

            rsp_length = ntohl(rsp_length); // net -> net endianness!

            cout << "Reading number of bytes: " << rsp_length << endl;

//            boost::asio::streambuf rbuf;
//            error_code r_ec;
//            boost::asio::read(*clientSocket, sbuf, boost::asio::transfer_exactly(rsp_length), &r_ec);
//            istream serializedRsp(&rbuf);

            uchar* rbuf = new uchar[rsp_length];

            boost::asio::read(*clientSocket, boost::asio::buffer(rbuf, rsp_length));

            LmkRsp rspMsg;
            rspMsg.ParseFromArray(rbuf, rsp_length);
            cout << "Lmk Size: " << rspMsg.dim().shape().size() << endl;
            cout << "Lmk Dim: " << rspMsg.dim().shape()[0] << ", " << rspMsg.dim().shape()[1] << endl;

            auto data = rspMsg.data();
            //set the matrix's data
            size_t dataSize = rspMsg.dim().shape()[0] *  rspMsg.dim().shape()[1];

            //allocate the matrix
            cv::Mat m(rspMsg.dim().shape()[0], rspMsg.dim().shape()[1],
                    CV_32F, data.data());
            cout << "M_lmks = "  << m << endl << endl;

            cv::cv2eigen(m, landmarks);
            cout << "eigen_Lmks:\n" <<  landmarks << endl;

            delete[] rbuf;
        }
        catch(exception& e) {
            cout << e.what() << endl;
        }

        return in;
    }

    bool FeatureDetectionClientPipe::connect(){
        isConnected = false;

        clientSocket = make_shared<SocketT>(ioService);

        try {
            clientSocket->connect(boost::asio::local::stream_protocol::endpoint(address));
//            ioService.restart();
            isConnected = true;
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            std::cerr << "Failed to connect to server..." << std::endl;
            disconnect();
        }
//
        return isConnected;
    }

    void FeatureDetectionClientPipe::disconnect(){
        isConnected = false;

        try {
            clientSocket->shutdown(boost::asio::ip::tcp::socket::shutdown_both);
            clientSocket->close();
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            std::cerr << "Failed to connect to server..." << std::endl;
        }

        clientSocket.reset();
    }

//    void FeatureDetectionClientPipe::set_result(optional<error_code>* a, error_code b)
//    {
//        a->reset(b);
//    }
//
//    bool FeatureDetectionClientPipe::read_with_timeout(boost::asio::streambuf& buffer)
//    {
//        optional<error_code> timer_result;
//        deadline_timer timer(clientSocket->get_io_service());
//        timer.expires_from_now(boost::posix_time::seconds(5));
//        timer.async_wait(boost::bind(&FeatureDetectionClientPipe::set_result, &timer_result, _1));
//
//        optional<error_code> read_result;
//        async_read(*clientSocket, buffer,
//                   boost::bind(&FeatureDetectionClientPipe::set_result, &read_result, _1));
//
//        clientSocket->get_io_service().reset();
//        while (clientSocket->get_io_service().run_one())
//        {
//            if (read_result)
//                timer.cancel();
//            else if (timer_result)
//                clientSocket->cancel();
//        }
//
//        if (*read_result)
//            throw system_error(*read_result);
//    }
}
