#include <string>
#include <pcl/io/image.h>
#include <cv.h>
#include <highgui.h>

#include "io/bmp.h"

namespace telef::io {
    void saveBMPFile(const std::string &path, unsigned char* rgb_buffer, unsigned int width, unsigned int height) {
        auto img = cv::Mat(width, height, CV_8UC3);

        memcpy(img.data, rgb_buffer, width*height*sizeof(uchar));

        cvtColor(img, img, CV_RGB2BGR);
        cv::imwrite(path, img);
    }

    void saveBMPFile(const std::string &path, const pcl::io::Image &image) {
        auto img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3);
        image.fillRGB(img.cols, img.rows, img.data, img.step);
        cvtColor(img, img, CV_RGB2BGR);
        cv::imwrite(path, img);
    }
}
