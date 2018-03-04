#include "io/OpenNI2Reader.h"
#include <OpenNI.h>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <chrono>
#include <thread>
#include "exceptions.h"

namespace {
    [[noreturn]] inline void putNIExtendedErrorAndDie(const std::string& msg)
    {
        throw openni::face::DeviceFailException(msg);
    }

    inline void requireStatus(openni::Status status, const std::string& errorMsg)
    {
        if (status != openni::STATUS_OK)
        {
            putNIExtendedErrorAndDie(errorMsg);
        }
    }

}

using namespace openni;
namespace openni::face {

    OpenNI2Reader::OpenNI2Reader() 
        :streams {&this->depthVideoStream, &this->colorVideoStream}
    {
        /* OpenNI Global Context Initialization */
        auto initStatus = OpenNI::initialize();
        requireStatus(initStatus, "OpenNI initialization failure");

        /* Open Any RGB-D Device. Init device */
        auto devOpenStatus = this->device.open(ANY_DEVICE);
        requireStatus(devOpenStatus, "Could not open device.");

        if (this->device.getSensorInfo(SENSOR_DEPTH) == NULL) 
        {
            putNIExtendedErrorAndDie("No depth sensor detected on your device");
        }
        if (this->device.getSensorInfo(SENSOR_COLOR) == NULL) 
        {
            putNIExtendedErrorAndDie("No color sensor detected on your device");
        }

        /* Init Video Stream */
        auto depthVideoStreamCreationStatus = this->depthVideoStream.create(this->device, SENSOR_DEPTH);
        requireStatus(depthVideoStreamCreationStatus, "Depth video stream creation failed");
        auto colorVideoStreamCreationStatus = this->colorVideoStream.create(this->device, SENSOR_COLOR);
        requireStatus(colorVideoStreamCreationStatus, "Color video stream creation failed");

        /* Start Reading Frames */
        auto colorStartStatus = this->colorVideoStream.start();
        requireStatus(colorStartStatus, "Could not start color stream");
        auto depthStartStatus = this->depthVideoStream.start();
        requireStatus(depthStartStatus, "Could not start depth stream");

        /* Set Registraion Mode On */
        if (!this->device.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR)) 
        {
            putNIExtendedErrorAndDie("Depth-to-color registration not supported.");
        }
        auto registraionInitStatus = this->device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
        requireStatus(registraionInitStatus, "Could not start depth-to-color registration");
    }

    OpenNI2Reader::~OpenNI2Reader() 
    {
        std::cout << "Closing" << std::endl;
        depthVideoStream.stop();
        colorVideoStream.stop();
        depthVideoStream.destroy();
        colorVideoStream.destroy();
        device.close();
        OpenNI::shutdown();
    }

    RGBDFrame OpenNI2Reader::syncReadFrame()
    {
        bool isColorFrameAcquired=false, isDepthFrameAcquired=false;
        RGBDFrame::SingleFramePtr colorFrame = std::make_shared<VideoFrameRef>();
        RGBDFrame::SingleFramePtr depthFrame = std::make_shared<VideoFrameRef>();

        while (!isColorFrameAcquired || !isDepthFrameAcquired)
        {
            int preparedStreamIndex = -1;
            auto waitStatus =
                OpenNI::waitForAnyStream((VideoStream**)this->streams, 2, &preparedStreamIndex, this->syncReadTimeOut);
            if (waitStatus != STATUS_OK) 
            {
                std::cout << "Wait failed. Retrying..." << std::endl;
                continue;
            }

            if(preparedStreamIndex == this->depthStreamIndex) 
            {
                auto readStatus = this->depthVideoStream.readFrame(depthFrame.get());
                if (readStatus != STATUS_OK) 
                {
                    std::cout << "Depth read failed. Retrying..." << std::endl;
                    continue;
                }
                isDepthFrameAcquired = true;
            }
            else if(preparedStreamIndex == this->colorStreamIndex) 
            {
                auto readStatus = this->colorVideoStream.readFrame(colorFrame.get());
                if (readStatus != STATUS_OK) 
                {
                    std::cout << "Color read failed. Retrying..." << std::endl;
                    continue;
                }
                isColorFrameAcquired = true;
            }
            else {
                std::cout << "Unexpected Stream" << std::endl;
            }
        }
        RGBDFrame frame{depthFrame, colorFrame};
        return frame;
    }
}

