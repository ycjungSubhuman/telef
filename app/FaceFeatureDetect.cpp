#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <sys/stat.h>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/io/image_rgb24.h>

#include "FaceAlignment.h"

#include "io/opencv_metadata_wrapper.h"
#include "feature/feature_detector.h"

namespace {
    using namespace std;
    using namespace telef::feature;
    using namespace telef::io;
}

void draw_box(cv::Mat &img, cv::Rect rect) {
	cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
}

bool fileExists(const string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

int test_cv(cv::Mat frame) {

    string detectionModel("../models/DetectionModel-v1.5.bin");
    string trackingModel("../models/TrackingModel-v1.10.bin");
    string faceDetectionModel("../models/haarcascade_frontalface_alt2.xml");

    cv::CascadeClassifier face_cascade;
    if (! face_cascade.load(faceDetectionModel))
    {
        cerr << "Error loading face detection model." << endl;
        return -1;
    }

    INTRAFACE::XXDescriptor xxd(4);
    auto fa = make_unique<INTRAFACE::FaceAlignment>(detectionModel.c_str(), trackingModel.c_str(), &xxd);

    if (! fa->Initialized()) {
        cerr << "FaceAlignment cannot be initialized." << endl;
        return -1;
    }


    if (!face_cascade.load(faceDetectionModel))
    {
        cerr << "Error loading face detection model." << endl;
        return -1;
    }

    vector<cv::Rect> faces;

    // face detection
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    face_cascade.detectMultiScale(frame_gray, faces, 1.2, 2, 0, cv::Size(50, 50));

    //INTRAFACE::HeadPose pose;
    float score, notFace = 0.5;

    cv::Mat featCoords;

    for (int i = 0; i < faces.size(); i++) {
        cv::Rect iface = faces[i];
        cout << faces[i] << endl;
        draw_box(frame, iface);
        if (fa->Detect(frame, faces[i], featCoords, score) == INTRAFACE::IF_OK)
        {
            // only draw valid faces
            if (score >= notFace) {
                cout << faces[i] << endl;
                for (int i = 0; i < featCoords.cols; i++) {
                    int x_coord = (int) featCoords.at<float>(0, i);
                    int y_coord = (int) featCoords.at<float>(1, i);
                    cv::circle(frame, cv::Point(x_coord, y_coord), 1, cv::Scalar(0, 255, 0), -1);
                }
            }
        }
    }

    imwrite("output.jpg", frame);

    return 0;
}

int test_detector(cv::Mat frame) {

    // Convert OpenCV Image to PCL Image
    pcl::io::FrameWrapper::Ptr imgWrapper(new OpencvFrameWrapper(frame));
    pcl::io::ImageRGB24 pclImg(imgWrapper);

    IntraFace featDetector;

    Feature featureData = featDetector.getFeature(pclImg);

    cout << "Feature Points: " << featureData.points << endl;

    for (int i = 0; i < featureData.points.cols(); ++i) {
        int x_coord = (int) featureData.points(0, i);
        int y_coord = (int) featureData.points(1, i);
        cv::circle(frame, cv::Point(x_coord, y_coord), 1, cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, std::to_string(i), cv::Point(x_coord+2, y_coord), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,0));
    }

    imwrite("output.jpg", frame);

    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: <image>";
        return -1;
    }
    string fname = argv[1];

    if (! fileExists(fname)) {
        cout << "File doesn't exist " << fname << endl;
    }
    else {
        cout << "Loading " << fname << "...\n";
    }

    cv::Mat frame;
    frame = cv::imread(fname, cv::IMREAD_COLOR);
    if (! frame.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    //return test_cv(frame);
    return test_detector(frame);
}

