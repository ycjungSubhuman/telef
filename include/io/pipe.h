#pragma once
#include <memory>
#include <functional>
#include <boost/shared_ptr.hpp>

namespace telef::io {
    // ---------------------------------------------------------
// "library" part
// ---------------------------------------------------------
    template<typename F1, typename F2>
    class Composite{
    private:
        F1  f1;
        F2  f2;

    public:
        Composite(F1  f1,  F2  f2) : f1(f1), f2(f2) { }

        template<typename IN>
        decltype(auto) operator() (IN i)
        {
            return f2( f1(i) );
        }
    };

// ---------------------------------------------------------
// ---------------------------------------------------------
    template<typename F1, typename F2>
    decltype(auto) compose (F1 f, F2 g) {
        return Composite<F1, F2> {f,g};
    }

// ---------------------------------------------------------
// ---------------------------------------------------------
    template<typename F1, typename... Fs>
    decltype(auto) compose (F1  f,  Fs  ... args)
    {
        return compose (f, compose(args...));
    }

    /**
     * A Step in Data Pipeline for Channel
     *
     * Can be constructed from a sequence of several Pipes.
     */
    template <class InT, class OutT>
    class Pipe {
    public:
        using FuncT = std::function<boost::shared_ptr<OutT>(boost::shared_ptr<InT>)>;

        boost::shared_ptr<OutT> operator()(boost::shared_ptr<InT> in) {
            return _processData(in);
        }
    private:
        virtual boost::shared_ptr<OutT> _processData(boost::shared_ptr<InT> in) = 0;
    };

    /**
     * A Simple Step That Does Nothing on The Input Data
     */
    template <class InT>
    class IdentityPipe : public Pipe<InT, InT> {
    private:
        boost::shared_ptr<InT> _processData(boost::shared_ptr<InT> in) override {
            return in;
        }
    };

}



