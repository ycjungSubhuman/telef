#include <iostream>
#include <experimental/filesystem>
#include "io/pipe.h"
#include "cloud/cloud_pipe.h"
#include "io/device.h"

namespace {
    namespace fs = std::experimental::filesystem;
    using namespace telef::io;
    using namespace telef::cloud;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
	 std::cout << "Usage : $ FakeDeviceTest <fake frame root dir>" << std::endl;
	 return 1;
    }

    std::string arg(argv[1]);
    fs::path recordRoot(arg);

    auto imagePipe = IdentityPipe<ImageT>();
    auto cloudPipe = RemoveNaNPoints();

    auto imageChannel = std::make_shared<DummyImageChannel<ImageT>>([&imagePipe](auto in)->decltype(auto){return imagePipe(in);});
    auto cloudChannel = std::make_shared<DummyCloudChannel<DeviceCloudConstT>>([&cloudPipe](auto in)->decltype(auto){return cloudPipe(in);});

    auto merger = std::make_shared<FittingSuiteMerger>();
    auto csvFrontend = std::make_shared<FittingSuiteWriterFrontEnd>(false, true, true);
    merger->addFrontEnd(csvFrontend);
    auto viewFrontend = std::make_shared<Landmark3DVisualizerFrontEnd>();
    merger->addFrontEnd(viewFrontend);

    FakeImagePointCloudDevice<DeviceCloudConstT, ImageT, FittingSuite, FittingSuite> device(recordRoot);
    device.setCloudChannel(cloudChannel);
    device.setImageChannel(imageChannel);
    device.addMerger(merger);

    device.run();
}
