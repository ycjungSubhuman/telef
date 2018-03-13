#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "FaceAlignment.h"
#include "XXDescriptor.h"

using namespace std;

bool drawing_box = false;
cv::Mat X;
cv::Rect box;

static cv::CascadeClassifier face_cascade;

void draw_box(cv::Mat &img, cv::Rect rect) {
	cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
}

bool fileExists(const string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

int main(int argc, char** argv) {
	cout << __cplusplus << endl;

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

	char *detectionModel = "../models/DetectionModel-v1.5.bin";
	char *trackingModel = "../models/TrackingModel-v1.10.bin";
	string faceDetectionModel("../models/haarcascade_frontalface_alt2.xml");

	cv::Mat frame;
	frame = cv::imread(fname, cv::IMREAD_COLOR);
	if (!frame.data) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	
	if (!face_cascade.load(faceDetectionModel))
	{
		cerr << "Error loading face detection model." << endl;
		return -1;
	}
	
    INTRAFACE::FaceAlignment *fa;
	INTRAFACE::XXDescriptor xxd(4);
	fa = new INTRAFACE::FaceAlignment(detectionModel, trackingModel, &xxd);
	
	if (!fa->Initialized()) {
		cerr << "FaceAlignment cannot be initialized." << endl;
		return -1;
	}

	vector<cv::Rect> faces;
	float score, notFace = 0.5;
	// face detection
	face_cascade.detectMultiScale(frame_gray, faces, 1.2, 2, 0, cv::Size(50, 50));

	INTRAFACE::HeadPose pose;

	for (int i = 0; i < faces.size(); i++) {
		cv::Rect iface = faces[i];
		cout << faces[i] << endl;
		draw_box(frame, iface);
		if (fa->Detect(frame, faces[i], X, score) == INTRAFACE::IF_OK)
		{
			// only draw valid faces
			if (score >= notFace) {
				cout << faces[i] << endl;
				for (int i = 0; i < X.cols; i++)
					cv::circle(frame, cv::Point((int)X.at<float>(0, i), (int)X.at<float>(1, i)), 1, cv::Scalar(0, 255, 0), -1);
			}
		}
	}

	imwrite("output.jpg", frame);

	delete fa;

	return 0;
}

