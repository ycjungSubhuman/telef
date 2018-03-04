#include <stdexcept>
#include <string>
#include <OpenNI.h>

namespace openni::face {
    class DeviceFailException: public std::runtime_error
    {
        public:
            DeviceFailException(std::string what)
                :std::runtime_error(what + "\n" + openni::OpenNI::getExtendedError()) {}
    };
    class FrameInterpretationException: public std::runtime_error
    {
        public:
            FrameInterpretationException(std::string what)
                :std::runtime_error(what) {}
    };
}
