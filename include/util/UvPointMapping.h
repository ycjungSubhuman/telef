#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

namespace telef::util {
    /**
     * Mapping from UV point ({0 ~ imageWidth}, {0 ~ imageHeight}) -> size_t
     */
    class UvPointMapping {
    public:
        UvPointMapping(int imageWidth, int imageHeight) {
            this->imageWidth = (size_t) imageWidth;
            this->imageHeight = (size_t) imageHeight;
            mapping = std::make_shared<std::vector<long>>(this->imageWidth * this->imageHeight, None);
        }

        /**
         * Add a single mapping from uv to point id
         */
        void addSingle(int u, int v, size_t pointId) {
            (*mapping)[v*imageWidth + u] = pointId;
        }

        /**
         * Add mapping change occured from filters i.e.) remove nan points
         *
         * @param mappingChange (processedPointCloud.points[i] = prevPointCloud[mappingChange[i]])
         *
         * This update is lazily evaluated for performance
         */
        void updateMapping(std::vector<int> mappingChange) {
            mappingChanges.emplace_back(mappingChange);
        }

        /**
         * Get Mapping
         */
        size_t getMappedPointId(int u, int v) {
            auto o = (*mapping)[v * imageWidth + u];
            if (o == None) {
                throw std::out_of_range("Mapping Does Not Exist");
            }
            for (const auto &mappingChange : mappingChanges) {
                auto processedIndex = std::find(mappingChange.begin(), mappingChange.end(), o) - mappingChange.begin();
                if (processedIndex >= mappingChange.size()) {
                    throw std::out_of_range("Mapping Does Not Exist");
                }
                o = processedIndex;
            }
            return (size_t) o;
        }

    private:
        static constexpr long None = -1;
        size_t imageWidth;
        size_t imageHeight;

        std::shared_ptr<std::vector<long>> mapping;
        std::vector<std::vector<int>> mappingChanges;
    };
}