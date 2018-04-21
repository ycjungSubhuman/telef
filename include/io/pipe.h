#pragma once
#include <memory>
#include <functional>
#include <boost/shared_ptr.hpp>

namespace telef::io {
    /**
     * A Step in Data Pipeline for Channel
     *
     * Can be constructed from a sequence of several Pipes.
     */
    template <class InT, class OutT>
    class Pipe {
    public:
        explicit Pipe(){
            this->processData = std::bind(&Pipe::_processData, this, std::placeholders::_1);
        }
        virtual ~Pipe() = default;
        Pipe& operator=(const Pipe&) = delete;
        Pipe(const Pipe&) = default;

        template<class NextOutT>
        std::shared_ptr<Pipe<InT, NextOutT>> then(std::shared_ptr<Pipe<OutT, NextOutT>> nextPipe) {
            auto nextProcessData = [*this,nextPipe=std::move(nextPipe)] (boost::shared_ptr<InT> in)-> boost::shared_ptr<NextOutT> {
                return nextPipe->processData(this->processData(in));
            };
            return std::shared_ptr<Pipe<InT, NextOutT>>{new Pipe{nextProcessData}};
        }

        using FuncT = std::function<boost::shared_ptr<OutT>(boost::shared_ptr<InT>)>;

        FuncT processData;
    private:
        // Default data process method
        virtual boost::shared_ptr<OutT> _processData(boost::shared_ptr<InT> in) {
            return boost::shared_ptr<OutT>();
        }
        explicit Pipe(const FuncT &processData) {
            this->processData = processData;
        }
    };

    /**
     * A Simple Step That Does Nothing on The Input Data
     */
    template <class InT>
    class IdentityPipe : public Pipe<InT, InT> {
    private:
        virtual boost::shared_ptr<InT> _processData(boost::shared_ptr<InT> in) {
            return in;
        }
    };

}



