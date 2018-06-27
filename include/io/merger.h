#pragma once

#include <boost/shared_ptr.hpp>
#include <memory>
#include <future>
#include <functional>
#include "type.h"
#include "io/pipe.h"
#include "feature/feature_detector.h"
#include "io/frontend.h"

namespace {
    using namespace telef::types;
    using namespace telef::feature;
}

namespace telef::io {

    /**
     * Merge Two Data into One OutT Data
     *
     * Further process OutT into PipeOutT using Pipe<OutT, PipeOutT>
     */
    template<class DataAT, class DataBT, class OutT, class PipeOutT>
    class BinaryMerger {
    private:
        using OutPtrT = const boost::shared_ptr<OutT>;
        using DataAPtrT = const boost::shared_ptr<DataAT>;
        using DataBPtrT = const boost::shared_ptr<DataBT>;
        using PipeOutPtrT = const boost::shared_ptr<PipeOutT>;
        using FuncT = std::function<boost::shared_ptr<PipeOutT>(boost::shared_ptr<OutT>)>;
    public:
        explicit BinaryMerger (FuncT pipe) {
            this->pipe = pipe;
        }
        virtual ~BinaryMerger() {
            for(auto &f : this->frontends) {
                f->stop();
            }
        }
        BinaryMerger& operator=(const BinaryMerger&) = delete;
        BinaryMerger (const BinaryMerger&) = default;

        void addFrontEnd(std::shared_ptr<FrontEnd<PipeOutT>> frontend) {
            this->frontends.push_back(frontend);
        }

        void run(DataAPtrT a, DataBPtrT b) {
            for (const auto& frontend : frontends) {
                frontend->process(getMergeOut(a, b));
            }
        }

    private:
        PipeOutPtrT getMergeOut(DataAPtrT a, DataBPtrT b) {
            auto merged = merge(a, b);
            return this->pipe(merged);
        }
        virtual OutPtrT merge(DataAPtrT a, DataBPtrT b)=0;
        FuncT pipe;
        std::vector<std::shared_ptr<FrontEnd<PipeOutT>>> frontends;
    };

    /**
     * Merge Two Data into One OutT Data without any further processing
     */
    template<class DataAT, class DataBT, class OutT>
    class SimpleBinaryMerger : public BinaryMerger<DataAT, DataBT, OutT, OutT> {
    private:
        using BaseT = BinaryMerger<DataAT, DataBT, OutT, OutT>;
        using OutPtrT = const boost::shared_ptr<OutT>;
        using DataAPtrT = const boost::shared_ptr<DataAT>;
        using DataBPtrT = const boost::shared_ptr<DataBT>;
    public:
        SimpleBinaryMerger() : BaseT([](auto in)->decltype(auto){return IdentityPipe<OutT>()(in);}) {}
        OutPtrT merge (DataAPtrT a, DataBPtrT b) override = 0;
    };

    /**
     * Merge const PointCloud and Image Using (UV coord) -> Point ID mapping
     */
    template<class OutT>
    class SimpleMappedImageCloudMerger : public SimpleBinaryMerger<ImageT, DeviceCloudConstT, OutT> {
    private:
        using OutPtrT = const boost::shared_ptr<OutT>;
        using MappedConstBoostPtrT = boost::shared_ptr<DeviceCloudConstT>;
    public:
        OutPtrT merge(const ImagePtrT image, const MappedConstBoostPtrT cloudPair) override=0;
    };

    // TODO: Inject landmark detectors later
    class FittingSuiteMerger : public SimpleMappedImageCloudMerger<FittingSuite> {
    private:
        using OutPtrT = const boost::shared_ptr<FittingSuite>;
        using DeviceCloudBoostPtrT = boost::shared_ptr<DeviceCloudConstT>;
    public:
        OutPtrT merge(const ImagePtrT image, const DeviceCloudBoostPtrT deviceCloud) override {
            auto landmark3d = boost::make_shared<CloudT>();
            auto rawCloud = deviceCloud->cloud;
            std::vector<int> rawCloudLmkIdx;
            auto mapping = deviceCloud->img2cloudMapping;
            feature::IntraFace featureDetector;
            auto feature = std::make_shared<Feature>(featureDetector.getFeature(*image));
            for (long i=0; i<feature->points.cols(); i++) {
                try {
                    auto pointInd = mapping->getMappedPointId(feature->points(0, i), feature->points(1, i));
                    landmark3d->push_back(rawCloud->at(pointInd));
                    rawCloudLmkIdx.push_back(pointInd);
                } catch (std::out_of_range &e) {
                    std::cout << "WARNING: Landmark Points at Hole." << std::endl;
                }
            }
            landmark3d->height = rawCloud->height;
            landmark3d->width = rawCloud->width;
            std::cout << landmark3d->size() <<std::endl;

            auto result = boost::make_shared<FittingSuite>();
            result->landmark2d = feature;
            result->landmark3d = landmark3d;
            result->rawCloud = rawCloud;
            result->rawCloudLmkIdx = rawCloudLmkIdx;
            result->rawImage = image;
            result->fx = deviceCloud->fx;
            result->fy = deviceCloud->fy;

            return result;
        }
    };

    // PipeOutT is the output of the given pipe, the pipe = FittingSuite -> pipe1 -> a -> pipe2 -> PipeOutT
    template <class PipeOutT>
    class FittingSuitePipeMerger : public BinaryMerger<ImageT, DeviceCloudConstT, FittingSuite, PipeOutT> {
    private:
        using BaseT = BinaryMerger<ImageT, DeviceCloudConstT, FittingSuite, PipeOutT>;
        using OutPtrT = const boost::shared_ptr<FittingSuite>;
        using DeviceCloudBoostPtrT = boost::shared_ptr<DeviceCloudConstT>;
        using FuncT = std::function<boost::shared_ptr<PipeOutT>(boost::shared_ptr<FittingSuite>)>;

    public:
        FittingSuitePipeMerger(FuncT pipe) : BaseT(pipe) {}
        OutPtrT merge(const ImagePtrT image, const DeviceCloudBoostPtrT deviceCloud) override {
            auto landmark3d = boost::make_shared<CloudT>();
            auto rawCloud = deviceCloud->cloud;
            std::vector<int> rawCloudLmkIdx;
            auto mapping = deviceCloud->img2cloudMapping;
            feature::IntraFace featureDetector;
            auto feature = std::make_shared<Feature>(featureDetector.getFeature(*image));
            auto badlmks = std::vector<int>();
            for (long i=0; i<feature->points.cols(); i++) {
                try {
                    auto pointInd = mapping->getMappedPointId(feature->points(0, i), feature->points(1, i));
                    landmark3d->push_back(rawCloud->at(pointInd));
                    rawCloudLmkIdx.push_back(pointInd);
                } catch (std::out_of_range &e) {
                    badlmks.push_back(i);
                }
            }

            if (badlmks.size() > 0) {
                std::cout << "WARNING: Landmark Points at Hole." << std::endl;
            }

            landmark3d->height = rawCloud->height;
            landmark3d->width = rawCloud->width;
            std::cout << "3D Lmks: " << landmark3d->size() <<std::endl;

            auto result = boost::make_shared<FittingSuite>();
            result->landmark2d = feature;
            result->landmark3d = landmark3d;
            result->invalid3dLandmarks = badlmks;
            result->rawCloud = rawCloud;
            result->rawCloudLmkIdx = rawCloudLmkIdx;
            result->rawImage = image;
            result->fx = deviceCloud->fx;
            result->fy = deviceCloud->fy;

            return result;
        }
    };

    /**
     * Just discard image and select PointCloud. Used for debugging
     */
    class DummyMappedImageCloudMerger : public SimpleMappedImageCloudMerger<CloudConstT> {
    private:
        using OutPtrT = const boost::shared_ptr<CloudConstT>;
        using DeviceCloudConstBoostPtrT = boost::shared_ptr<DeviceCloudConstT>;
    public:
        OutPtrT merge(const ImagePtrT image, const DeviceCloudConstBoostPtrT deviceCloud) override {
            return deviceCloud->cloud;
        }
    };
}

