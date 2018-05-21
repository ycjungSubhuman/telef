#pragma once
#include <memory>
#include "io/pipe.h"
#include "face/classified_model.h"

namespace telef::face {

    class ClassifiedFittingSuiteT {
    public:
        std::shared_ptr<telef::face::MorphableFaceModel> model;
        boost::shared_ptr<telef::feature::FittingSuite> fittingSuite;
    };

    class ClassifyMorphableModelPipe :
            public telef::io::Pipe<telef::feature::FittingSuite, ClassifiedFittingSuiteT>{
    private:
        using InT = ClassifiedFittingSuiteT;
    public:
        using OutPtrT = boost::shared_ptr<InT>;
        ClassifyMorphableModelPipe(ClassifiedMorphableModel model)
        : model(model) {}
    private:
        ClassifiedMorphableModel model;

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

    class ClassifiedRigidFittingPipe : public telef::io::Pipe<ClassifiedFittingSuiteT, telef::align::PCARigidAlignmentSuite> {
    private:
        boost::shared_ptr<telef::align::PCARigidAlignmentSuite>
        _processData(boost::shared_ptr<ClassifiedFittingSuiteT> in ) override {

            telef::align::PCARigidFittingPipe rigidPipe(in->model);
            return rigidPipe(in->fittingSuite);
        }
    };
}