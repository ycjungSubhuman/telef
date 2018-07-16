#pragma once

#include <dlib/dnn.h>

#include "io/pipe.h"
#include "feature/face.h"


namespace {
    using namespace dlib;
}

namespace telef::feature {

    /**
     * Face Detection using Open CV haar cascade
     */
    class FaceDetectionPipe : public telef::io::Pipe<telef::io::DeviceInputSuite, FeatureDetectSuite> {
    protected:
        using BaseT = telef::io::Pipe<telef::io::DeviceInputSuite, Feature>;
        using InputPtrT = telef::io::DeviceInputSuite::Ptr;
    };

    /**
     * Face Detection using Open CV haar cascade
     */
//    class HaarFaceDetectionPipe : public FaceDetectionPipe {
//    private:
//        using BaseT = FaceDetectionPipe::BaseT;
//        using InputPtrT = FaceDetectionPipe::InputPtrT;
//
//        FaceDetectionSuite::Ptr _processData(InputPtrT in) override;
//
//    public:
//        HaarFaceDetectionPipe(const std::string &pretrained_model, const bool faceProfile);
//    };

    /**
     * Face Detection using Dlib CNN, Not realtime
     */
    class DlibFaceDetectionPipe : public FaceDetectionPipe {
    private:
        using BaseT = FaceDetectionPipe::BaseT;
        using InputPtrT = FaceDetectionPipe::InputPtrT;

        template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
        template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
        template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
        template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

        using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

        net_type net;

        //TODO: Keep old face location incase CNN fails???

        FeatureDetectSuite::Ptr _processData(InputPtrT in) override;

    public:
        DlibFaceDetectionPipe(const std::string &pretrained_model);
    };
}
