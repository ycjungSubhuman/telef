#include "RGBDFrame.h"
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <OpenNI.h>
#include <vector>
#include <cassert>
#include <cstdlib>
#include "exceptions.h"

using namespace openni::face;
using namespace openni;

namespace {
    pcl::PointXYZRGB getXYZRGB(const DepthPixel& depthPixel, const RGB888Pixel& colorPixel)
    {
        pcl::PointXYZRGB p;
        p.x = 0;
        p.y = 0;
        p.z = 0;
        p.rgb = 0;
        return p;
    }

    std::vector<pcl::PointXYZRGB> frameToPoints(RGBDFrame::SingleFramePtr depth, RGBDFrame::SingleFramePtr color) {

        auto depthPixelFormat = depth->getVideoMode().getPixelFormat();
        if (depthPixelFormat != PIXEL_FORMAT_DEPTH_1_MM && depthPixelFormat != PIXEL_FORMAT_DEPTH_100_UM) 
        {
            throw FrameInterpretationException("Depth pixel format not supported");
        }
        auto colorPixelFormat = color->getVideoMode().getPixelFormat();
        if (colorPixelFormat != PIXEL_FORMAT_RGB888)
        {
            throw FrameInterpretationException("Color pixel format not supported");
        }
        if (depth->getWidth() != color->getWidth() || depth->getHeight() != color->getHeight())
        {
            throw FrameInterpretationException("Color and Depth have different shapes");
        }
    
        std::vector<pcl::PointXYZRGB> result;
        DepthPixel* depthData = (DepthPixel*)depth->getData();
        RGB888Pixel* colorData = (RGB888Pixel*)color->getData();

        int pixelCount = depth->getWidth() * depth->getHeight();

        for (int i=0; i<pixelCount; i++)
        {
            DepthPixel depthPixel = depthData[i];
            RGB888Pixel colorPixel = colorData[i];
            result.push_back(getXYZRGB(depthPixel, colorPixel));
        }

        return result;
    }
}

namespace openni::face {

    RGBDFrame::RGBDFrame(RGBDFrame::SingleFramePtr depth, RGBDFrame::SingleFramePtr color)
    {
        this->depth = depth;
        this->color = color;
    }

    pcl::PointCloud<pcl::PointXYZRGB> RGBDFrame::toPointCloud() const
    {
        std::cout << "Stub toPointCloud" << std::endl;
        auto points = frameToPoints(this->depth, this->color);
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        for (auto&& p : points) {
            cloud.push_back(p);
        }
        return cloud;
    }
}

