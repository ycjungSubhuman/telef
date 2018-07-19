#pragma once

#include <boost/shared_ptr.hpp>
#include <memory>
#include <future>
#include <functional>
#include "type.h"
#include "io/pipe.h"
#include "feature/feature_detector.h"
#include "io/frontend.h"
#include "io/merger.h"
#include "io/merger/device_input.h"

namespace {
    using namespace telef::types;
}

namespace telef::io {
    // PipeOutT is the output of the given pipe, the pipe = FittingSuite -> pipe1 -> a -> pipe2 -> PipeOutT
    template<class PipeOutT>
    class DeviceInputPipeMerger : public telef::io::BinaryMerger<ImageT, DeviceCloudConstT, DeviceInputSuite, PipeOutT> {
    private:
        using BaseT = BinaryMerger<ImageT, DeviceCloudConstT, DeviceInputSuite, PipeOutT>;
        using OutPtrT = const DeviceInputSuite::Ptr;
        using DeviceCloudBoostPtrT = boost::shared_ptr<DeviceCloudConstT>;
        using FuncT = std::function<boost::shared_ptr<PipeOutT>(DeviceInputSuite::Ptr)>;

    public:
        DeviceInputPipeMerger(FuncT pipe) : BaseT(pipe) {}

        OutPtrT merge(const ImagePtrT image, const DeviceCloudBoostPtrT deviceCloud) override {
            auto result = boost::make_shared<DeviceInputSuite>();
            result->rawCloud = deviceCloud->cloud;
            result->rawImage = image;
            result->img2cloudMapping = deviceCloud->img2cloudMapping;
            result->fx = deviceCloud->fx;
            result->fy = deviceCloud->fy;

            return result;
        }
    };
}
