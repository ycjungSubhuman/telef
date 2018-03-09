#include "io/pipe.h"

namespace telef::io {
    template<class InT, class OutT>
    Pipe<InT, OutT>::Pipe() {
        this->processData = std::bind(&Pipe::_processData, this, std::placeholders::_1);
    }

    template<class InT, class OutT>
    template<class NextOutT>
    Pipe<InT, NextOutT> Pipe<InT, OutT>::then(std::unique_ptr<Pipe<OutT, NextOutT>> nextPipe) {
        auto nextProcessData = [this,nextPipe{std::move(nextPipe)}] (InT in)->NextOutT {
            return nextPipe->processData(this->processData(in));
        };
        return Pipe{std::move(nextProcessData)};
    }

    template<class InT, class OutT>
    Pipe<InT, OutT>::Pipe(std::function<OutT(InT)> processData) {
        this->processData = std::move(processData);
    }
}