#pragma once

#include <boost/function.hpp>
#include <memory>
#include <mutex>
#include <pcl/io/image.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "io/pipe.h"
#include "type.h"

namespace {
using namespace telef::types;
}

namespace telef::io {
/**
 * Data Channel for Device.
 */
template <class DataT, class OutDataT> class Channel {
public:
  // Use boost shared_ptr for pcl compatibility
  using DataPtrT = boost::shared_ptr<DataT>;
  using OutDataPtrT = boost::shared_ptr<OutDataT>;
  using FuncT =
      std::function<boost::shared_ptr<OutDataT>(boost::shared_ptr<DataT>)>;

  explicit Channel(FuncT pipe) {
    this->grabberCallback = boost::bind(&Channel::_grabberCallback, this, _1);
    this->pipe = std::move(pipe);
  }
  Channel(const Channel &) = delete;

  /**
   * Called in Deveice run() Loop
   */
  OutDataPtrT onDeviceLoop() {
    DataPtrT data;
    this->currentData.swap(data);
    if (data) {
      auto outData = this->pipe(data);
      this->onOutData(outData);
      return outData;
    } else {
      return OutDataPtrT();
    }
  }

  /**
   * Callback to be registerd to pcl::Grabber
   */
  boost::function<void(const DataPtrT &)> grabberCallback;

protected:
  /**
   * Handle data according to channel usage.
   *
   * This function is called before it enters any Merger.
   */
  virtual void onOutData(OutDataPtrT data) = 0;

private:
  DataPtrT currentData;
  FuncT pipe;

  void _grabberCallback(const DataPtrT &fetchedInstance) {
    this->currentData = fetchedInstance;
  }
};

/**
 * Fetch XYZRGBA Point Cloud from OpenNI2 Devices
 */
template <class OutDataT>
class CloudChannel : public Channel<DeviceCloudConstT, OutDataT> {
public:
  using FuncT = std::function<boost::shared_ptr<OutDataT>(
      boost::shared_ptr<DeviceCloudConstT>)>;
  explicit CloudChannel(FuncT pipe)
      : Channel<DeviceCloudConstT, OutDataT>(std::move(pipe)) {}

protected:
  void onOutData(boost::shared_ptr<OutDataT> data) override = 0;
};

/**
 * Fetch RGB Image from OpenNI2 Devices
 */
template <class OutDataT>
class ImageChannel : public Channel<ImageT, OutDataT> {
public:
  using FuncT =
      std::function<boost::shared_ptr<OutDataT>(boost::shared_ptr<ImageT>)>;
  explicit ImageChannel(FuncT pipe)
      : Channel<ImageT, OutDataT>(std::move(pipe)) {}

protected:
  void onOutData(boost::shared_ptr<OutDataT> data) override = 0;
};

template <class OutDataT>
class DummyCloudChannel : public CloudChannel<OutDataT> {
public:
  using FuncT = std::function<boost::shared_ptr<OutDataT>(
      boost::shared_ptr<DeviceCloudConstT>)>;
  DummyCloudChannel()
      : CloudChannel<OutDataT>([](auto in) -> decltype(auto) { return in; }) {}
  explicit DummyCloudChannel(FuncT pipe)
      : CloudChannel<OutDataT>(std::move(pipe)) {}

protected:
  void onOutData(boost::shared_ptr<OutDataT> data) override {}
};

template <class OutDataT>
class DummyImageChannel : public ImageChannel<OutDataT> {
public:
  using FuncT =
      std::function<boost::shared_ptr<OutDataT>(boost::shared_ptr<ImageT>)>;
  DummyImageChannel()
      : ImageChannel<OutDataT>([](auto in) -> decltype(auto) { return in; }) {}
  explicit DummyImageChannel(FuncT pipe)
      : ImageChannel<OutDataT>(std::move(pipe)) {}

protected:
  void onOutData(boost::shared_ptr<OutDataT> data) override {}
};

} // namespace telef::io