

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

#include "util/eigen_file_io.h"
#include "util/pcl_cv.h"
#include "feature/feature_detect_pipe.h"

using namespace std;
using namespace dlib;
using namespace telef::io;


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
        //    load_image(img, argv[i]);

        // Convert PCL image to dlib image
        auto pclImage = in->rawImage;

        auto matImg = cv::Mat(pclImage->getHeight(), pclImage->getWidth(), CV_8UC3);
        pclImage->fillRGB(matImg.cols, matImg.rows, matImg.data, matImg.step);
        cv::cvtColor(matImg, matImg, CV_RGB2BGR);
        dlib::cv_image<bgr_pixel> cvImg(matImg);
        dlib::assign_image(img, cvImg);


        // Downsampling the image will allow us to use real time, but user will have to be very close to camera
//        pyramid_down<2> myDownSampler;
//        while (img.size() > 256 * 256)
//            myDownSampler(img);
       
        // Note that you can process a bunch of images in a std::vector at once and it runs
        // much faster, since this will form mini-batches of images and therefore get
        // better parallelism out of your GPU hardware.  However, all the images must be
        // the same size.  To avoid this requirement on images being the same size we
        // process them individually in this example.
        auto begin = chrono::high_resolution_clock::now();
        auto dets = net(img);
        auto end = chrono::high_resolution_clock::now();
        auto dur = end - begin;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        cout << "Face Detection(ms):" << ms << endl;

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
            : lmkDetector(graphPath, checkpointPath), prnetIntputSize(256)

    {
    }

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

        // Takes 3x3 Matrix, square for inverse
//        cv::Mat P = cv::Mat::eye(3,3,CV_64F);
//        transform.copyTo(P.rowRange(0,2));
//        cv::warpPerspective(normImage, warped, P, cv::Size(dst_size, dst_size) );

    }

    void PRNetFeatureDetectionPipe::restore(Eigen::MatrixXf& restored, const Eigen::MatrixXf& result, const cv::Mat& transform){
        cv::Mat resultMat; //(result.rows(), result.cols(), transform.type());
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
            std::cout << "Face Detection Failed, returning previous landmarks...";
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

        //warped Lmk Result
        cv::Mat warpedCpy;
        warped.convertTo(warpedCpy, CV_8UC3);


        int myradius=3;
        for (int i=0;i<result.cols();i++)
            cv::circle(warpedCpy,cvPoint(result(0,i),result(1,i)),myradius,CV_RGB(0,255,0),-1,8,0);

        cv::imshow("Warped Landmarks", warpedCpy);

        // Result
        cv::Mat imgCpy;
        matImg->copyTo(imgCpy);

        for (int i=0;i<landmarks.cols();i++)
            cv::circle(imgCpy,cvPoint(landmarks(0,i),landmarks(1,i)),myradius,CV_RGB(0,255,0),-1,8,0);

        cv::imshow("Landmarks", imgCpy);
        cv::waitKey(0);

        in->feature->points = landmarks;
        return in;
    }
}