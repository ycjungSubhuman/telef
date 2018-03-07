#pragma once

#include <pcl/io/grabber.h>
#include "io/fetcher.h"

namespace telef::io {

    /** Interface with pcl::Grabber */
    class Device {
    public:
        Device(pcl::Grabber* grabber);

        template <class FetchedInstanceT>
        void addFetcher(Fetcher<FetchedInstanceT> fetcher);
    };
}