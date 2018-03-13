#pragma once

#include "type.h"
#include "io/pipe.h"
#include "feature/feature_detector.h"

using namespace telef::feature;
using namespace telef::types;
using namespace telef::io;

namespace telef::image {
    class DummyFeatureDetectorPipe : public Pipe<ImageT, Feature> {
        virtual boost::shared_ptr<Feature> _processData(boost::shared_ptr<ImageT> in) override {
            auto feature = boost::make_shared<Feature>();
            feature->points.resize(5, 2);
            feature->points <<
                            320, 190,
                            310, 190,
                            315, 195,
                            315, 200,
                            310, 210;

            return feature;
        }
    };
}
