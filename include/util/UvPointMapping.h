#pragma once

#include <experimental/filesystem>

namespace {
    namespace fs = std::experimental::filesystem;
}

namespace telef::util {
    /**
     * Mapping from UV point ({0 ~ imageWidth}, {0 ~ imageHeight}) -> size_t
     */
    // TODO: Remove This class and use internal structure of pcl::PointCloud.
    // This approach is only useful when we use unstructured point cloud
    class UvPointMapping {
    private:
        size_t uvToInd(int u, int v);

        int IndToU(size_t ind);

        int IndToV(size_t ind);

    public:
        UvPointMapping() = default;
        UvPointMapping(int imageWidth, int imageHeight);
        UvPointMapping(fs::path p);

        /**
         * Add a single mapping from uv to point id
         */
        void addSingle(int u, int v, size_t pointId);

        /**
         * Add mapping change occured from filters i.e.) remove nan points
         *
         * @param mappingChange (processedPointCloud.points[i] = prevPointCloud[mappingChange[i]])
         *
         * This update is lazily evaluated for performance
         */
        void updateMapping(std::vector<int> mappingChange);

        /**
         * Get Mapping
         */
        size_t getMappedPointId(int u, int v);

        void forceApplyChanges();

        void save(fs::path p);

    private:
        static constexpr long None = -1;
        size_t imageWidth;
        size_t imageHeight;

        std::shared_ptr<std::vector<long>> mapping;
        std::vector<std::vector<int>> mappingChanges;
    };
}