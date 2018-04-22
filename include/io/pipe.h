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
        using FuncT = std::function<boost::shared_ptr<OutT>(boost::shared_ptr<InT>)>;
        FuncT composed;

        explicit Pipe() {composed=[](boost::shared_ptr<InT> in){return boost::shared_ptr<OutT>();};}
        virtual ~Pipe() = default;
        Pipe& operator=(const Pipe&) = delete;
        Pipe(const Pipe&) = default;

        explicit Pipe(const FuncT &processData) {
            this->composed = processData;
        }

        template<class NextOutT>
        std::shared_ptr<Pipe<InT, NextOutT>> then(std::shared_ptr<Pipe<OutT, NextOutT>> nextPipe) {
            auto nextProcessData = [curr=*this, nextPipe=std::move(nextPipe)] (boost::shared_ptr<InT> in)-> boost::shared_ptr<NextOutT> {
                return nextPipe->composed(curr.composed(in));
            };
            return std::shared_ptr<Pipe<InT, NextOutT>>{new Pipe<InT, NextOutT>{nextProcessData}};
        }
    };

    /**
     * A Simple Step That Does Nothing on The Input Data
     */
    template <class InT>
    class IdentityPipe : public Pipe<InT, InT> {
    public:
        IdentityPipe() {
            this->composed = std::bind(&IdentityPipe::_processData, this, std::placeholders::_1);
        }
    private:

        boost::shared_ptr<InT> _processData(boost::shared_ptr<InT> in) {
            return in;
        }
    };

}



