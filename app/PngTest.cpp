#include <iostream>

#include "io/png.h"
#include "type.h"

namespace {
    using namespace telef::io;
    using namespace telef::types;
    namespace fs = std::experimental::filesystem;
}

int main(int argc, char**argv)
{
    if (argc <= 1) return 1;

    std::string str(argv[1]);
    auto image = loadPNG(str);

    std::cout << image->getWidth() << "/" <<image->getHeight() << std::endl;

    savePNG("pngtest.png", *image);

    return 0;
}