#include "util/po_util.h"

namespace {
    namespace po = boost::program_options;
}

namespace telef::util {
    void require(const po::variables_map &vm, const std::string argname) {
        if (vm.count(argname) == 0) {
            std::cout << "Please specify " << argname << std::endl;
            std::cout << "Give '--help' option to see all available options" << argname << std::endl;
            exit(1);
        }
    }
}