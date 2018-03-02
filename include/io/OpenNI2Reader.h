#pragma once
#include <OpenNI.h> 
#include "RGBDFrame.h"

using namespace openni;
namespace openni::face {

    /**
     * Reader for RGBD Stream captured through OpenNI2
     *
     *
     * Don't Use this class with other modules using OpenNI.
     * This class uses global context of OpenNI.
     */
    class OpenNI2Reader {
        private:
            const int syncReadTimeOut = 2000; //ms
            VideoStream depthVideoStream, colorVideoStream;
            VideoStream* streams[2];
            const int depthStreamIndex = 0; // defined by depthVideoStream's position in streams
            const int colorStreamIndex = 1; // defined by colorVideoStream's position in streams
            Device device;

        public:
            OpenNI2Reader();
            ~OpenNI2Reader();

            RGBDFrame syncReadFrame();
            // TODO : Add async mode for future performance
    };
}


