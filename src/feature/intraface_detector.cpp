#include "feature/feature_detector.h"

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "FaceAlignment.h"

using namespace std;

namespace telef::feature {

    Feature IntraFace::getFeature(const FeatureDetector::ImageT &image) {
        Feature feature;

        //TODO: Add configuration file (XML) and Configuration Object.
        // Need to store hard coded values in a configuration
        string detectionModel("../models/DetectionModel-v1.5.bin");
        string trackingModel("../models/TrackingModel-v1.10.bin");
        string faceDetectionModel("../models/haarcascade_frontalface_alt2.xml");

        // Convert PCL image to OpenCV Mat
        auto imgrgb = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3);
        auto img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3);
        image.fillRGB(imgrgb.cols, imgrgb.rows, imgrgb.data, imgrgb.step);
        cvtColor(imgrgb, img, CV_RGB2BGR);

        // Initialize Face and Feature Detectors
        // TODO: Store detectors as member, to not need instantiation each time used
        // TODO: Decouple Face Detection with Feature Detection?
        // TODO: Try also using face profile detector to detect sides of face
        cv::CascadeClassifier face_cascade;
        if (! face_cascade.load(faceDetectionModel))
        {
            cerr << "Error loading face detection model." << endl;
            return Feature();
        }

        INTRAFACE::XXDescriptor xxd(4);
        auto fa = make_unique<INTRAFACE::FaceAlignment>(detectionModel.c_str(), trackingModel.c_str(), &xxd);

        if (! fa->Initialized()) {
            cerr << "FaceAlignment cannot be initialized." << endl;
            return Feature();
        }

        // Detect Face
        // Gray scale needed for Cascade Face Detector
        cv::Mat img_gray;
        cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        vector<cv::Rect> faces;
        face_cascade.detectMultiScale(img_gray, faces, 1.2, 2, 0, cv::Size(50, 50));

        //INTRAFACE::HeadPose pose;

        // Detect Features from detected faces
        float score, notFace = 0.5;
        for (int i = 0; i < faces.size(); i++) {
            cv::Rect iface = faces[i];

            cv::Mat featCoords;
            if (fa->Detect(img, iface, featCoords, score) == INTRAFACE::IF_OK) {
                // Only get valid faces
                if (score >= notFace) {
                    // Set feature (Return Object)
                    feature.setBoundingBox(iface);
                    cv::cv2eigen(featCoords, feature.points);

                    // Only get first valid face
                    return feature;
                }
            }
        }

        return feature;
    }
}

