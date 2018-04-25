#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <experimental/filesystem>
#include <boost/shared_ptr.hpp>

#include "face/model.h"
#include "align/rigid_pipe.h"

namespace fs = std::experimental::filesystem;

namespace telef::face {
    /** Select from a collection of morphable model */
    template <int ShapeRank>
    class ClassifiedMorphableModel {
    public:
        using InPairT = std::pair<std::string, std::vector<fs::path>>;
        using ModelPairT = std::pair<std::string, MorphableFaceModel<ShapeRank>>;

        using MeshPairT = std::pair<std::string, Eigen::VectorXf>;
        ClassifiedMorphableModel(std::vector<InPairT> paths) {
            std::transform(paths.begin(), paths.end(), models.begin(), [](const auto & p)->decltype(auto) {
                return std::make_pair(p.first, MorphableFaceModel<ShapeRank>(p.second));
            });
        }

        /**
         * Get closest morphable face model to scan.
         *
         * Uses mean maximum distance from point samples in scanCloud to mesn meshes in morpahble models as the measure
         * of closeness
         *
         * The reason we first sample from scanCloud is we want to sample points of frontal face, and scanCloud has only
         * frontal part of scanned face. It includes some backgrounds or body, too. But we can cut them out by rejecting
         * them with some distance threshold.
         */
        MorphableFaceModel<ShapeRank> getClosestModel(boost::shared_ptr<telef::feature::FittingSuite> fittingSuite) {
            std::vector<double> dists;

            std::transform(models.begin(), models.end(), dists.begin(), [fittingSuite](const auto &p)->decltype(auto) {
                auto name = p.first;
                // We first align mean mesh of current MM with the scan
                auto mean = p.second.genPosition(Eigen::VectorXf::Zero(ShapeRank));
                telef::align::PCARigidFittingPipe rigidPipe(p.second);
                auto alignment = rigidPipe(fittingSuite);
                mean.applyTransform(alignment->transformation);

                auto meshCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                meshCloud->resize(mean.size());
                for(unsigned long i=0; i<mean.size()/3; i++) {
                    meshCloud->points[3*i] = mean[3*i];
                    meshCloud->points[3*i+1] = mean[3*i+1];
                    meshCloud->points[3*i+2] = mean[3*i+2];
                }

                // Find the closest point for each point in scanned cloud
                pcl::KdTreeFLANN kdtree;
                kdtree.setInputCloud(meshCloud);

                double distSum = 0.0f;
                int totalNearestPoint = 0;
                for(unsigned long i=0; i<fittingSuite->rawCloud->points.size(); i++) {
                    auto point = fittingSuite->rawCloud->points[i];

                    std::vector<int> nearestPointInds;
                    std::vector<float> nearestPointDists;
                    auto numPointsFound = kdtree.nearestKSearch(point,1, nearestPointInds, nearestPointDists);

                    if(numPointsFound == 1 && nearestPointDists[0] < REJECTION_THRESHOLD) {
                        distSum += nearestPointDists[0];
                        totalNearestPoint++;
                    }
                }

                if(totalNearestPoint==0) totalNearestPoint = 1;

                std::cout << "name : " << name << " dist : " << distSum / totalNearestPoint << std ::endl;
                return distSum / totalNearestPoint;
            });

            auto min = std::min_element(dists.begin(), dists.end());
            auto argmin = std::distance(dists.begin(), min);

            std::cout << "Selection : " << models[argmin].first << std ::endl;

            return models[argmin].second;
        }

    private:
        std::vector<ModelPairT> models;
        constexpr static const float REJECTION_THRESHOLD = 0.001f;
    };
}
