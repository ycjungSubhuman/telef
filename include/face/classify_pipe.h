#pragma once
#include <memory>
#include "io/pipe.h"
#include "face/classified_model.h"

namespace telef::face {

    template <int ShapeRank>
    class ClassifiedFittingSuiteT {
    public:
        std::shared_ptr<telef::face::MorphableFaceModel<ShapeRank>> model;
        boost::shared_ptr<telef::feature::FittingSuite> fittingSuite;
    };

    template <int ShapeRank>
    class ClassifyMorphableModelPipe :
            public telef::io::Pipe<telef::feature::FittingSuite, ClassifiedFittingSuiteT<ShapeRank>>{
    private:
        using InT = ClassifiedFittingSuiteT<ShapeRank>;
    public:
        using OutPtrT = boost::shared_ptr<InT>;
        ClassifyMorphableModelPipe(ClassifiedMorphableModel<ShapeRank> model)
        : model(model) {}
    private:
        ClassifiedMorphableModel<ShapeRank> model;

        OutPtrT
        _processData(boost::shared_ptr<telef::feature::FittingSuite> in) override
        {
            auto selectedModel = model.getClosestModel(in);
            auto result = boost::make_shared<InT>();
            result->model = selectedModel;
            result->fittingSuite = in;
            return result;
        }
    };

    template <int ShapeRank>
    class ClassifiedRigidFittingPipe : public telef::io::Pipe<ClassifiedFittingSuiteT<ShapeRank>, telef::align::PCARigidAlignmentSuite> {
    private:
        boost::shared_ptr<telef::align::PCARigidAlignmentSuite>
        _processData(boost::shared_ptr<ClassifiedFittingSuiteT<ShapeRank>> in ) override {

            telef::align::PCARigidFittingPipe rigidPipe(in->model);
            return rigidPipe(in->fittingSuite);
        }
    };
}