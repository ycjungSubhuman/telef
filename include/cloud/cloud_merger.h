#pragma once

#include "feature/feature_detector.h"
#include "io/merger.h"
#include "type.h"

using namespace telef::io;
using namespace telef::types;
using namespace telef::feature;

namespace telef::cloud {
    class LandmarkMerger : public SimpleBinaryMerger<CloudConstT, Feature, CloudConstT> {
    private:
        const boost::shared_ptr<CloudConstT>
        merge(const boost::shared_ptr<CloudConstT> cloud, const boost::shared_ptr<Feature> landmark) override {
            return cloud;
        }
    };
}