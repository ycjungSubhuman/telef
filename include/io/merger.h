#pragma once

#include <boost/shared_ptr.hpp>
#include "type.h"
#include "io/pipe.h"

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

        PipeOutPtrT getMergeOut(DataAPtrT a, DataBPtrT b) {
            auto merged = merge(a, b);
            return this->pipe->processData(merged);
        }
    private:
        virtual OutPtrT merge(DataAPtrT a, DataBPtrT b)=0;
        std::shared_ptr<PipeT> pipe;
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
     * Merge const PointCloud and Image
     *
     * Just discard image and select PointCloud. Used for debugging
     */
    class DummyImageCloudMerger : public SimpleBinaryMerger<CloudConstT, ImageT, CloudConstT> {
    private:
        using PipeT = Pipe<CloudConstT, CloudConstT>;
        using BaseT = SimpleBinaryMerger<ImageT, CloudConstT, CloudConstT>;
        using OutPtrT = const boost::shared_ptr<CloudConstT>;
    public:
        OutPtrT merge(const CloudConstPtrT cloud, const ImagePtrT image) override {
            return cloud;
        }
    };
}

