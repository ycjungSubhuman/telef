#include <string>
#include <pcl/io/image.h>
#include <opencv2/opencv.hpp>

#include "io/bmp.h"

namespace telef::io {
    void saveBMPFile(const std::string &path, const pcl::io::Image &image) {
        auto img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3);
        image.fillRGB(img.cols, img.rows, img.data, img.step);
        cvtColor(img, img, CV_RGB2BGR);
        cv::imwrite(path, img);
    }
}
