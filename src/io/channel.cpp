#include <boost/bind.hpp>
#include "io/channel.h"
#include <iostream>

namespace telef::io {

    void CloudChannel::onData(CloudChannel::DataPtrT data) {
        std::cout << "CloudChannel OnData: " << data->size() << std::endl;
    }

    void ImageChannel::onData(ImageChannel::DataPtrT data) {
        std::cout << "ImageChannel OnData: ("
                  << data->getWidth()
                  << "/" << data->getHeight()
                  << ")" << std::endl;
    }
}
