#include <iostream>

#include "intrinsic/intrinsic.h"
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
//#include <opencv2/imgproc.hpp>

namespace {
//using namespace telef::io;
//using namespace telef::types;
//using namespace telef::cloud;
using namespace telef::intrinsic;
} // namespace
/**
 * Continuously get frames of point cloud and image.
 *
 * Prints size of pointcloud and size of the image on every frame received
 * Remove all points that have NaN Position on Receive.
 * You can check this by watching varying number of pointcloud size
 */

int main(int ac, char *av[]) {
  /*pcl::io::OpenNI2Grabber::Mode depth_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  pcl::io::OpenNI2Grabber::Mode image_mode =
      pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

  auto grabber = new TelefOpenNI2Grabber("#1", depth_mode, image_mode);

  auto imagePipe = IdentityPipe<ImageT>();
  auto cloudPipe = IdentityPipe<DeviceCloudConstT>();
  auto cloudPipe2 = RemoveNaNPoints();
  auto cloudPipe3 = IdentityPipe<DeviceCloudConstT>();
  auto cloudCombinedPipe = compose(cloudPipe, cloudPipe2, cloudPipe3);
  auto func = [&cloudCombinedPipe](auto in) -> decltype(auto) {
    return cloudCombinedPipe.operator()(in);
  };

  auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>(
      [&imagePipe](auto in) -> decltype(auto) { return imagePipe(in); });
  auto cloudChannel =
      std::make_shared<DummyCloudChannel<DeviceCloudConstT>>(func);

  auto frontend = std::make_shared<DummyCloudFrontEnd>();
  auto merger = std::make_shared<DummyMappedImageCloudMerger>();
  merger->addFrontEnd(frontend);

  ImagePointCloudDeviceImpl<DeviceCloudConstT, ImageT, CloudConstT, CloudConstT>
      device{std::move(grabber)};
  device.setCloudChannel(cloudChannel);
  device.setImageChannel(imageChannel);
  device.addMerger(merger);

  device.run();*/

  std::cout << cv::imread << std::endl;

  //cv::Mat3b rgb(cv::imread("../tests/resources/0.png"));
  //cv::Mat depth(cv::imread("../tests/resources/0.d.png",CV_LOAD_IMAGE_ANYDEPTH));
  //cv::Mat normal(cv::imread("../tests/resources/0.n.png"));
  //cv::Mat1f intensity(640,480);

  //IntrinsicDecomposition ID;

  //ID.initialize(rgb.data,normal.data,depth.data,640,480);
  //ID.process(intensity.data)
  //ID.release();

  return 0;
}
