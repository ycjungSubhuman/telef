#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <boost/make_shared.hpp>
#include <experimental/filesystem>
#include <pcl/kdtree/kdtree_flann.h>
#include <string>
#include <vector>

#include "align/rigid_pipe.h"
#include "face/model.h"
#include "feature/feature_detector.h"

namespace {
namespace fs = std::experimental::filesystem;
}

namespace telef::face {
/** Select from a collection of morphable model */
// TODO: Test this class
class ClassifiedMorphableModel {
public:
  using ModelPairT = struct ModelPairT {
    std::string name;
    std::shared_ptr<MorphableFaceModel> model;
  };
  using InPairT = std::pair<std::string, fs::path>;

  ClassifiedMorphableModel(std::vector<InPairT> &paths) {
    for (unsigned long i = 0; i < paths.size(); i++) {
      ModelPairT result;
      result.name = std::string(paths[i].first);
      result.model = std::make_shared<MorphableFaceModel>(paths[i].second);
      models.push_back(result);
    }
  }

  /**
   * Get closest morphable face model to scan.
   *
   * Uses mean maximum distance from point samples in scanCloud to mesn meshes
   * in morpahble models as the measure of closeness
   *
   * The reason we first sample from scanCloud is we want to sample points of
   * frontal face, and scanCloud has only frontal part of scanned face. It
   * includes some backgrounds or body, too. But we can cut them out by
   * rejecting them with some distance threshold.
   */
  // TODO: Model multimodal distribution with more accurate method
  std::shared_ptr<MorphableFaceModel> getClosestModel(
      boost::shared_ptr<telef::feature::FittingSuite> fittingSuite) {
    std::vector<double> dists;

    for (unsigned long i = 0; i < models.size(); i++) {
      auto p = models[i];
      auto name = p.name;
      // We first align mean mesh of current MM with the scan
      auto meanMesh = models[i].model->genMesh(
          Eigen::VectorXf::Zero(models[i].model->getShapeRank()),
          Eigen::VectorXf::Zero(models[i].model->getExpressionRank()));
      telef::align::PCARigidFittingPipe rigidPipe(p.model);
      auto alignment = rigidPipe(fittingSuite);
      meanMesh.applyTransform(alignment->transformation);
      auto mean = meanMesh.position;

      auto meshCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      meshCloud->resize(mean.size());
      for (unsigned long i = 0; i < mean.size() / 3; i++) {
        meshCloud->points[3 * i].x = mean[3 * i];
        meshCloud->points[3 * i].y = mean[3 * i + 1];
        meshCloud->points[3 * i].z = mean[3 * i + 2];
      }

      // Find the closest point for each point in scanned cloud
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      kdtree.setInputCloud(meshCloud);

      double distSum = 0.0f;
      int totalNearestPoint = 0;
      for (unsigned long i = 0; i < fittingSuite->rawCloud->points.size();
           i++) {
        auto xyzrgb = fittingSuite->rawCloud->points[i];
        pcl::PointXYZ point;
        point.x = xyzrgb.x;
        point.y = xyzrgb.y;
        point.z = xyzrgb.z;

        std::vector<int> nearestPointInds;
        std::vector<float> nearestPointDists;
        auto numPointsFound = kdtree.nearestKSearch(point, 1, nearestPointInds,
                                                    nearestPointDists);

        if (numPointsFound == 1 && nearestPointDists[0] < REJECTION_THRESHOLD) {
          distSum += nearestPointDists[0];
          totalNearestPoint++;
        }
      }

      if (totalNearestPoint == 0)
        totalNearestPoint = 1;

      std::cout << "name : " << name
                << " dist : " << distSum / totalNearestPoint << std ::endl;
      dists.push_back(distSum / totalNearestPoint);
    }

    auto min = std::min_element(dists.begin(), dists.end());
    auto argmin = std::distance(dists.begin(), min);

    std::cout << "Selection : " << models[argmin].name << std ::endl;

    return models[argmin].model;
  }

private:
  std::vector<ModelPairT> models;
  constexpr static const float REJECTION_THRESHOLD = 0.001f;
};
} // namespace telef::face
