#pragma once

#include "cloud/cloud_pipe.h"
#include "io/channel.h"
#include "io/pipe.h"
#include "type.h"

namespace {
using namespace telef::io;
using namespace telef::types;
} // namespace
namespace telef::cloud {
/** Remove NaN Positioned Points from PointCloud */
class RemoveNaNPoints : public Pipe<DeviceCloudConstT, DeviceCloudConstT> {
private:
  boost::shared_ptr<DeviceCloudConstT>
  _processData(boost::shared_ptr<DeviceCloudConstT> in) override;
};

/** Remove NaN Positioned Points from PointCloud */
class FastBilateralFilterPipe
    : public Pipe<DeviceCloudConstT, DeviceCloudConstT> {
private:
  using BaseT = Pipe<DeviceCloudConstT, DeviceCloudConstT>;
  /** standard deviation of the Gaussian used by the bilateral filter for
   * the spatial neighborhood/window. (size of the Gaussian bilateral filter
   * window to use)
   */
  float sigma_s;

  /** standard deviation of the Gaussian used to control how much an adjacent
   * pixel is downweighted because of the intensity difference (depth in our
   * case). (standard deviation of the Gaussian for the depth difference)
   */
  float sigma_r;

  boost::shared_ptr<DeviceCloudConstT>
  _processData(boost::shared_ptr<DeviceCloudConstT> in) override;

public:
  FastBilateralFilterPipe(float sigma_s = 15, float sigma_r = 5e-2);
};
} // namespace telef::cloud
