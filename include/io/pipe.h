#pragma once
#include <memory>
#include <functional>

#include "io/channel.h"

namespace telef::io {
    /**
     * Data Pipeline for Channel
     */
    template <class InT, class OutT>
    class Pipe {
    public:
        Pipe();

        template<class NextOutT>
        Pipe<InT, NextOutT> then(std::unique_ptr<Pipe<OutT, NextOutT>> nextPipe);

        std::function<OutT(InT)> processData;
    private:
        // Default data process method
        virtual OutT _processData(InT in)=0;

        explicit Pipe(std::function<OutT(InT)> processData);
    };

}



