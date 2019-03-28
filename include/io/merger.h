#pragma once

#include "feature/face.h"
#include "feature/feature_detector.h"
#include "io/frontend.h"
#include "io/pipe.h"
#include "type.h"
#include <boost/shared_ptr.hpp>
#include <functional>
#include <future>
#include <memory>

namespace {
using namespace telef::types;
using namespace telef::feature;
} // namespace

namespace telef::io {

/**
 * Merge Two Data into One OutT Data
 *
 * Further process OutT into PipeOutT using Pipe<OutT, PipeOutT>
 */
template <class DataAT, class DataBT, class OutT, class PipeOutT>
class BinaryMerger {
private:
  using OutPtrT = const boost::shared_ptr<OutT>;
  using DataAPtrT = const boost::shared_ptr<DataAT>;
  using DataBPtrT = const boost::shared_ptr<DataBT>;
  using PipeOutPtrT = const boost::shared_ptr<PipeOutT>;
  using FuncT =
      std::function<boost::shared_ptr<PipeOutT>(boost::shared_ptr<OutT>)>;

public:
  explicit BinaryMerger(FuncT pipe) { this->pipe = pipe; }
  virtual ~BinaryMerger() {
    for (auto &f : this->frontends) {
      f->stop();
    }
  }
  BinaryMerger &operator=(const BinaryMerger &) = delete;
  BinaryMerger(const BinaryMerger &) = default;

  void addFrontEnd(std::shared_ptr<FrontEnd<PipeOutT>> frontend) {
    this->frontends.push_back(frontend);
  }

  void run(DataAPtrT a, DataBPtrT b) {
    for (const auto &frontend : frontends) {
      frontend->process(getMergeOut(a, b));
    }
  }

private:
  PipeOutPtrT getMergeOut(DataAPtrT a, DataBPtrT b) {
    auto merged = merge(a, b);
    return this->pipe(merged);
  }
  virtual OutPtrT merge(DataAPtrT a, DataBPtrT b) = 0;
  FuncT pipe;
  std::vector<std::shared_ptr<FrontEnd<PipeOutT>>> frontends;
};

/**
 * Merge Two Data into One OutT Data without any further processing
 */
template <class DataAT, class DataBT, class OutT>
class SimpleBinaryMerger : public BinaryMerger<DataAT, DataBT, OutT, OutT> {
private:
  using BaseT = BinaryMerger<DataAT, DataBT, OutT, OutT>;
  using OutPtrT = const boost::shared_ptr<OutT>;
  using DataAPtrT = const boost::shared_ptr<DataAT>;
  using DataBPtrT = const boost::shared_ptr<DataBT>;

public:
  SimpleBinaryMerger()
      : BaseT([](auto in) -> decltype(auto) {
          return IdentityPipe<OutT>()(in);
        }) {}
  OutPtrT merge(DataAPtrT a, DataBPtrT b) override = 0;
};

/**
 * Merge const PointCloud and Image Using (UV coord) -> Point ID mapping
 */
template <class OutT>
class SimpleMappedImageCloudMerger
    : public SimpleBinaryMerger<ImageT, DeviceCloudConstT, OutT> {
private:
  using OutPtrT = const boost::shared_ptr<OutT>;
  using MappedConstBoostPtrT = boost::shared_ptr<DeviceCloudConstT>;

public:
  OutPtrT merge(const ImagePtrT image,
                const MappedConstBoostPtrT cloudPair) override = 0;
};

/**
 * Just discard image and select PointCloud. Used for debugging
 */
class DummyMappedImageCloudMerger
    : public SimpleMappedImageCloudMerger<CloudConstT> {
private:
  using OutPtrT = const boost::shared_ptr<CloudConstT>;
  using DeviceCloudConstBoostPtrT = boost::shared_ptr<DeviceCloudConstT>;

public:
  OutPtrT merge(const ImagePtrT image,
                const DeviceCloudConstBoostPtrT deviceCloud) override {
    return deviceCloud->cloud;
  }
};
} // namespace telef::io
