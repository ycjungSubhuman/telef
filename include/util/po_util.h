#pragma once
#include <boost/program_options.hpp>
#include <string>
#include <iostream>

namespace {
    namespace po = boost::program_options;
}

namespace telef::util {
    void require(const po::variables_map &vm, const std::string argname);
}
