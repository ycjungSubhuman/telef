#pragma once

#include <vector>
#include <exception>

#include "mesh/mesh.h"

using namespace telef::mesh;

namespace telef::cluster {
    template<class T>
    class Cluster {
        virtual std::vector<std::vector<T>> cluster(std::vector<T> data) = 0;
    };

    template<class T>
    class KMeansCluster : Cluster<T> {
    public:
        KMeansCluster(int numCluster) : numCluster(numCluster) {

        }
        virtual std::vector<std::vector<T>> cluster(std::vector<T> data) {

        }

    private:
        virtual float distance(T &a, T &b) const { throw std::runtime_error("Not Implemented distance");}
        int numCluster;

        void expectation() {

        }
        void maximization() {

        }
    };

    template<>
    float KMeansCluster<ColorMesh>::distance(ColorMesh &a, ColorMesh &b) const {
        auto d = a.position - b.position;
        return d.norm();
    }
}