#pragma once

#include <fstream>
#include <sys/stat.h>
#include <time.h>

#include "boost/shared_ptr.hpp"

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/ml/ml.hpp>

#include <experimental/filesystem>

namespace telef::util {
namespace {
namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;
} // namespace

bool fileExists(const string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

boost::shared_ptr<cv::Mat> readCSVtoCV(string fname) {
  if (!fileExists(fname)) {
    throw std::runtime_error("File does not exist");
  }

  CvMLData mlData;
  mlData.read_csv(fname.c_str());

  // cv::Mat takes ownership of this pointer
  const CvMat *tmp = mlData.get_values();
  boost::shared_ptr<cv::Mat> cvMat(new cv::Mat(tmp, true));

  return cvMat;
}

/**
 * Reads a CSV file an returns the data as an Eigan Matrix
 *
 * @param fname
 * @param rows
 * @param cols
 * @return
 */
boost::shared_ptr<Eigen::MatrixXf> readCSVtoEigan(string fname) {

  // Using OpenCV to be able to create an arbitrary sized matrix
  // as well as not reinventing the wheel.
  boost::shared_ptr<cv::Mat> cvsMat = readCSVtoCV(fname);

  boost::shared_ptr<Eigen::MatrixXf> eigMat(new Eigen::MatrixXf());
  cv::cv2eigen(*cvsMat, *eigMat);

  return eigMat;
}
} // namespace telef::util
