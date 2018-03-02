#include <iostream>
#include "io/OpenNI2Reader.h"
#include "RGBDFrame.h"

using namespace openni::face;

int main(int ac, char* av[]) 
{
    if(ac <= 1) 
    {
        std::cerr << "Please specify output path as the first argument(eg : ./KinnectToPLY ./wow.ply)" << std::endl;
        return 1;
    }

    std::string outputPath {av[1]};

    OpenNI2Reader reader;
    RGBDFrame frame = reader.syncReadFrame();

    auto a = frame.toPointCloud();

    return 0;
}

