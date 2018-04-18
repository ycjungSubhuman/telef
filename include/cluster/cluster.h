#pragma once

#include <vector>
#include <exception>
#include <random>
#include <algorithm>
#include <memory>

#include "mesh/mesh.h"

using namespace telef::mesh;

namespace telef::cluster {
    template<class T>
    class Cluster {
        virtual std::vector<int> cluster(std::vector<T> &data) = 0;
    };

    template<class T>
    class KMeansCluster : Cluster<T> {
    public:
        KMeansCluster(int numCluster, int maxIteration=100)
                : numCluster(numCluster),
                  maxIteration(maxIteration),
                  mt(rd()),
                  centers(static_cast<unsigned long>(numCluster))
        {
        }
        virtual std::vector<int> cluster(std::vector<T> &data) {
            groups.resize(data.size());
            // Initialize centers with random points
            std::uniform_int_distribution<> dist {0, static_cast<int>(data.size() - 1)};
            for (int i=0; i<numCluster; i++) {
                centers[i] = data[dist(mt)];
            }

            // Initialize groups
            assignGroups(data);

            for (int i=0; i<maxIteration; i++) {
                estimateCenters(data);
                assignGroups(data);
            }

            return groups;
        }

    private:
        float distance(const T &a, const T &b) const;
        T mean(const std::vector<std::shared_ptr<T>> &a) const;
        int numCluster;
        int maxIteration;
        std::random_device rd;
        std::mt19937 mt;
        std::vector<int> groups;
        std::vector<T> centers;

        void assignGroups(std::vector<T> &data) {
            for (unsigned long i=0; i<data.size(); i++) {
                std::vector<float> distances;
                std::transform(centers.begin(), centers.end(), distances.begin(),
                               [i, &data, this](const auto& c){return this->distance(c, data[i]);});
                auto minElemIterator = std::min_element(distances.begin(), distances.end());
                groups[i] = static_cast<int>(std::distance(distances.begin(), minElemIterator));
            }
        }
        void estimateCenters(std::vector<T> &data) {
            std::vector<std::vector<std::shared_ptr<T>>> groupedData(static_cast<unsigned long>(numCluster));
            for (unsigned int i=0; i<data.size(); i++) {
                groupedData[groups[i]].push_back(std::shared_ptr<T>(&data[i]));
            }
            for (int i=0; i<numCluster; i++) {
                centers[i] = mean(groupedData[i]);
            }
        }
    };

    template<>
    float KMeansCluster<ColorMesh>::distance(const ColorMesh &a, const ColorMesh &b) const {
        auto d = a.position - b.position;
        return d.norm();
    }

    template<>
    ColorMesh KMeansCluster<ColorMesh>::mean(const std::vector<std::shared_ptr<ColorMesh>> &data) const {
        if(data.empty()) {
            return ColorMesh();
        }

        Eigen::VectorXf position(data[0]->position.rows());
        for (const auto &d : data) {
            position += d->position;
        }

        ColorMesh result;
        result.position = position / data.size();
        result.triangles = data[0]->triangles;
        return result;
    }
}