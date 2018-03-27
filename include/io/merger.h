#pragma once

#include <boost/shared_ptr.hpp>
#include <memory>
#include "type.h"
#include "io/pipe.h"
#include "feature/feature_detector.h"
#include "io/frontend.h"

using namespace telef::types;

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
        using PipeT = Pipe<OutT, PipeOutT>;
    public:
        explicit BinaryMerger (std::shared_ptr<PipeT> pipe) {
            this->pipe = pipe;
        }
        virtual ~BinaryMerger() = default;
        BinaryMerger& operator=(const BinaryMerger&) = delete;
        BinaryMerger (const BinaryMerger&) = default;

        void addFrontEnd(std::shared_ptr<FrontEnd<PipeOutT>> frontend) {
            this->frontends.emplace_back(frontend);
        }

        void run(DataAPtrT a, DataBPtrT b) {
            for (const auto& frontend : frontends) {
                frontend->process(getMergeOut(a, b));
            }
        }

    private:
        PipeOutPtrT getMergeOut(DataAPtrT a, DataBPtrT b) {
            auto merged = merge(a, b);
            return this->pipe->processData(merged);
        }
        virtual OutPtrT merge(DataAPtrT a, DataBPtrT b)=0;
        std::shared_ptr<PipeT> pipe;
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
        SimpleBinaryMerger() : BaseT(std::make_shared<IdentityPipe<OutT>>()) {}
        OutPtrT merge (DataAPtrT a, DataBPtrT b) override = 0;
    };

    /**
     * Merge const PointCloud and Image Using (UV coord) -> Point ID mapping
     */
    template<class OutT>
    class SimpleMappedImageCloudMerger : public SimpleBinaryMerger<ImageT, MappedCloudConstT, OutT> {
    private:
        using OutPtrT = const boost::shared_ptr<OutT>;
        using MappedConstBoostPtrT = boost::shared_ptr<MappedCloudConstT>;
    public:
        OutPtrT merge(const ImagePtrT image, const MappedConstBoostPtrT cloudPair) override=0;
    };

    class LandmarkMerger : public SimpleMappedImageCloudMerger<CloudConstT> {
    private:
        using OutPtrT = const boost::shared_ptr<CloudConstT>;
        using MappedConstBoostPtrT = boost::shared_ptr<MappedCloudConstT>;
    public:
        OutPtrT merge(const ImagePtrT image, const MappedConstBoostPtrT cloudPair) override {
            auto result = boost::make_shared<CloudT>();
            auto cloud = cloudPair->first;
            auto mapping = cloudPair->second;
            feature::IntraFace featureDetector;
            auto feature = featureDetector.getFeature(*image);
            for (long i=0; i<feature.points.cols(); i++) {
                try {
                    auto pointInd = mapping->getMappedPointId(feature.points(0, i), feature.points(1, i));
                    result->push_back(cloud->at(pointInd));
                } catch (std::out_of_range &e) {
                    std::cout << "WARNING: Landmark Points at Hole." << std::endl;
                }
            }
            result->height = cloud->height;
            result->width = cloud->width;
            std::cout << result->size() <<std::endl;
            return result;
        }
    };

    /**
     * Just discard image and select PointCloud. Used for debugging
     */
    class DummyMappedImageCloudMerger : public SimpleMappedImageCloudMerger<CloudConstT> {
    private:
        using OutPtrT = const boost::shared_ptr<CloudConstT>;
        using MappedConstBoostPtrT = boost::shared_ptr<MappedCloudConstT>;
    public:
        OutPtrT merge(const ImagePtrT image, const MappedConstBoostPtrT cloud) override {
            return cloud->first;
        }
    };
}

