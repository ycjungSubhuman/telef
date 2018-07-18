#include <thread>
#include <chrono>
#include "vis/fitting_visualizer.h"

using namespace telef::vis;

int main() {
    auto a = FittingVisualizer();

    std::this_thread::sleep_for(std::chrono::seconds(10));
    return 0;
}
