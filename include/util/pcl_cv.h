#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/image.h>

namespace telef::util {
boost::shared_ptr<cv::Mat> convert(boost::shared_ptr<pcl::io::Image> pclImage) {

  auto matImg = boost::make_shared<cv::Mat>(pclImage->getHeight(),
                                            pclImage->getWidth(), CV_8UC3);
  pclImage->fillRGB(matImg->cols, matImg->rows, matImg->data, matImg->step);
  cv::cvtColor(*matImg, *matImg, CV_RGB2BGR);

  return matImg;
}
} // namespace telef::util