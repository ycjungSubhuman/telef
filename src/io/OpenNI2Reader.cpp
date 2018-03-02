#include "io/OpenNI2Reader.h"
#include <OpenNI.h>
#include <iostream>
#include <memory>
#include <cstdlib>

namespace {
    [[noreturn]] inline void putNIExtendedErrorAndDie(const std::string& msg) 
    {
        std::cerr << msg << std::endl;
        std::cerr << OpenNI::getExtendedError() << std::endl;
        exit(EXIT_FAILURE);
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
        requireStatus(initStatus, "OpenNI Initialization Failure");

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

        /* Start reading frames */
        auto colorStartStatus = this->colorVideoStream.start();
        requireStatus(colorStartStatus, "Could not start color stream");
        auto depthStartStatus = this->depthVideoStream.start();
        requireStatus(depthStartStatus, "Could not start depth stream");
    }

    OpenNI2Reader::~OpenNI2Reader() 
    {
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
        std::shared_ptr<VideoFrameRef> colorFrame=std::make_shared<VideoFrameRef>();
        std::shared_ptr<VideoFrameRef> depthFrame=std::make_shared<VideoFrameRef>();

        while (!isColorFrameAcquired && !isDepthFrameAcquired)
        {
            int preparedStream = -1;
            OpenNI::waitForAnyStream((VideoStream**)this->streams, 2, &preparedStream, this->syncReadTimeOut);

            if(preparedStream == this->depthStreamIndex) 
            {
                isDepthFrameAcquired = true;
                this->depthVideoStream.readFrame(depthFrame.get());
            }
            else if(preparedStream == this->colorStreamIndex) 
            {
                isColorFrameAcquired = true;
                this->colorVideoStream.readFrame(colorFrame.get());
            }
            else {
                std::cout << "Unexpected Stream" << std::endl;
            }
        }
        RGBDFrame frame{depthFrame, colorFrame};
        return frame;
    }
}

