#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include "type.h"

using namespace telef::types;

namespace telef::io {
    /**
     * Do Something with Side Effect Provided InputT
     *
     * Used as final step in Device
     */
    template<class InputT>
    class FrontEnd {
    private:
        using InputPtrT = const boost::shared_ptr<InputT>;
    public:
        virtual ~FrontEnd() = default;
        virtual void process(InputPtrT input)=0;
    };


    class DummyCloudFrontEnd : public FrontEnd<CloudConstT> {
    private:
        using InputPtrT = const CloudConstPtrT;
    public:
        void process(InputPtrT input) override {
            std::cout << "DummyCloudFrontEnd : " << input->size() << std::endl;
        }
    };
}
