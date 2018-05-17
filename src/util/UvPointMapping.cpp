#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <assert.h>

#include "util/UvPointMapping.h"

namespace {
    namespace fs = std::experimental::filesystem;
}

namespace telef::util {
    UvPointMapping::UvPointMapping(int imageWidth, int imageHeight) {
        this->imageWidth = (size_t) imageWidth;
        this->imageHeight = (size_t) imageHeight;
        mapping = std::make_shared<std::vector<long>>(this->imageWidth * this->imageHeight, None);
    }

    UvPointMapping::UvPointMapping(fs::path p) {
        std::ifstream f(p, std::ios_base::binary);
        size_t len;

        f.read((char*)(&imageWidth), sizeof(size_t));
        f.read((char*)(&imageHeight), sizeof(size_t));
        mapping = std::make_shared<std::vector<long>>(imageWidth * imageHeight);
        f.read((char*)mapping->data(), (imageWidth*imageHeight)*sizeof(long));
    }

    size_t UvPointMapping::uvToInd(int u, int v) {
        return v*imageWidth + u;
    }

    int UvPointMapping::IndToU(size_t ind) {
        return static_cast<int>(ind % imageWidth);
    }

    int UvPointMapping::IndToV(size_t ind) {
        return static_cast<int>(ind / imageWidth);
    }

    void UvPointMapping::addSingle(int u, int v, size_t pointId) {
        (*mapping)[uvToInd(u, v)] = pointId;
    }

    void UvPointMapping::updateMapping(std::vector<int> mappingChange) {
        mappingChanges.emplace_back(mappingChange);
    }

    size_t UvPointMapping::getMappedPointId(int u, int v) {
        auto o = (*mapping)[uvToInd(u, v)];
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

    void UvPointMapping::forceApplyChanges() {
        for (size_t i=0; i<mapping->size(); i++) {
            int u = IndToU(i);
            int v = IndToV(i);

            auto o = (*mapping)[uvToInd(u, v)];
            if (o == None) {
                continue;
            }
            for (const auto &mappingChange : mappingChanges) {
                auto processedIndex = std::find(mappingChange.begin(), mappingChange.end(), o) - mappingChange.begin();

                if (processedIndex < mappingChange.size()) {
                    (*mapping)[i] = processedIndex;
                }
            }
        }
        mappingChanges.clear();
    }

    void UvPointMapping::save(fs::path p) {
        forceApplyChanges();
        std::ofstream f(p, std::ios_base::binary);

        size_t mappingSize = mapping->size();

        f.write((char*)(&imageWidth), sizeof(size_t));
        f.write((char*)(&imageHeight), sizeof(size_t));
        f.write((char*)(&mappingSize), sizeof(size_t));
        assert(mappingSize == imageWidth*imageHeight);

        f.write((char*)mapping->data(), mapping->size()*sizeof(long));
        f.close();
    }

}
