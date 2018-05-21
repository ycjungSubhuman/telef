#include <iostream>
#include <experimental/filesystem>
#include <sys/time.h>

#include "io/ply/meshio.h"
#include "face/model.h"

namespace {
    namespace fs = std::experimental::filesystem;
}

decltype(auto) intv (timeval &t1, timeval &t2) {
    return t2.tv_sec - t1.tv_sec +
    (t2.tv_usec - t1.tv_usec) / 1.0e6;
}

int main() {
    telef::face::MorphableFaceModel model(fs::path("data/example"));
    telef::face::MorphableFaceModel modelgpu(fs::path("data/example"), true);

    std::vector<telef::mesh::ColorMesh> wow;

    timeval cpuStart, cpuEnd, gpuStart, gpuEnd;
    gettimeofday(&cpuStart, NULL);
    for(int i=1; i<=100; i++) {
        wow.push_back(model.sample());
    }
    gettimeofday(&cpuEnd, NULL);

    gettimeofday(&gpuStart, NULL);
    for(int i=1; i<=100; i++) {
        wow.push_back(modelgpu.sample());
    }
    gettimeofday(&gpuEnd, NULL);

    std::cout << "CPU : " << intv(cpuStart, cpuEnd) << "s" << " GPU: " << intv(gpuStart, gpuEnd) << "s" << std::endl;

    for(auto &m : wow) {
        telef::io::ply::writePlyMesh(fs::path("temp.ply"), m);
    }

    return 0;
}