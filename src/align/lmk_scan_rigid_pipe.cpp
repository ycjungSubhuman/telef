#include <iostream>

#include <pcl/common/copy_point.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>

#include "align/rigid_pipe.h"

using namespace std;
using namespace telef::feature;
using namespace telef::types;
using namespace telef::face;

namespace telef::align {

LmkToScanRigidFittingPipe::LmkToScanRigidFittingPipe()
    : BaseT(), transformation(Eigen::Matrix4f::Identity(4, 4)),
      landmark3d(new pcl::PointCloud<pcl::PointXYZRGBA>()) {}

boost::shared_ptr<FittingSuite>
LmkToScanRigidFittingPipe::_processData(FeatureDetectSuite::Ptr in) {
  // Only update transform and landmarks if features were found
  auto feature = in->feature;

  std::vector<int> rawCloudLmkIdx;
  auto badlmks = std::vector<int>();
  if (feature->points.size() > 0) {
    // Find Correspondences between Image(Feature points) and Scan
    // & Determine valid and invalid lmks
    auto rawCloud = in->deviceInput->rawCloud;
    auto mapping = in->deviceInput->img2cloudMapping;
    auto prnetCorr = boost::make_shared<CloudT>();
    auto scanCorr = boost::make_shared<CloudT>();

    std::vector<int> rigidCorr;
    int validLmkCount = 0;
    for (long i = 0; i < feature->points.cols(); i++) {
      try {
        auto pointInd = mapping->getMappedPointId(
            feature->points(0, i), feature->points(1, i));
        auto scanPoint = rawCloud->at(pointInd);
        scanCorr->push_back(scanPoint);
        rawCloudLmkIdx.push_back(pointInd);

        pcl::PointXYZRGBA lmkPoint;
        pcl::copyPoint(scanPoint, lmkPoint);
        lmkPoint.x = feature->points(0, i);
        lmkPoint.y = feature->points(1, i);
        lmkPoint.z = feature->points(2, i);
        prnetCorr->push_back(lmkPoint);

        // Only do rigid fitting on most rigid landmarks (exclude chin for scan
        // alignment)
        if (i > 16) {
          rigidCorr.push_back(validLmkCount);
        }
        validLmkCount++;
      } catch (std::out_of_range &e) {
        badlmks.push_back(i);
      }
    }

    // Align Lmks to Scan
    Eigen::Matrix4f currentTransform;

    // Fill correspondance list with range 0, ..., n valid landmarks.
    pcl::registration::
        TransformationEstimationSVDScale<pcl::PointXYZRGBA, pcl::PointXYZRGBA>
            svd;
    svd.estimateRigidTransformation(
        *prnetCorr, rigidCorr, *scanCorr, rigidCorr, currentTransform);

    // Return Aligned 3D Landmarks via PointCloud
    auto transformed_lmks = boost::make_shared<CloudT>();
    // You can either apply transform_1 or transform_2; they are the same
    pcl::transformPointCloud(*prnetCorr, *transformed_lmks, currentTransform);

    transformation = currentTransform;
    Eigen::MatrixXf finalTransformedLmks =
        transformation * (in->feature->points.colwise().homogeneous().matrix());
    CloudPtrT lmk3d = boost::make_shared<CloudT>();
    lmk3d->resize(static_cast<size_t>(finalTransformedLmks.cols()));
    for (int i = 0; i < finalTransformedLmks.cols(); i++) {
      lmk3d->points[i].x = finalTransformedLmks(0, i);
      lmk3d->points[i].y = finalTransformedLmks(1, i);
      lmk3d->points[i].z = finalTransformedLmks(2, i);
    }
    landmark3d.swap(lmk3d);
  } else {
    std::cout << "No Landmarks detected, using previous frames...\n";
  };

  boost::shared_ptr<FittingSuite> output = boost::make_shared<FittingSuite>();
  output->landmark3d = landmark3d;
  output->landmark2d.swap(feature);
  output->invalid3dLandmarks = badlmks;
  output->rawCloudLmkIdx = rawCloudLmkIdx;
  output->rawImage.swap(in->deviceInput->rawImage);
  output->rawCloud = in->deviceInput->rawCloud;
  output->fx = in->deviceInput->fx;
  output->fy = in->deviceInput->fy;

  return output;
}
} // namespace telef::align