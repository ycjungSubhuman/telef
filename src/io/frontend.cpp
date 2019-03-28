#include <iostream>
#include <pcl/io/ply_io.h>

#include "io/devicecloud.h"
#include "io/frontend.h"
#include "io/png.h"
#include "type.h"

namespace {
namespace fs = std::experimental::filesystem;
using namespace telef::types;
using namespace telef::util;
using namespace telef::io;
} // namespace

namespace telef::io {
bool FittingSuiteWriterFrontEnd::checkProcessNeeded(
    FittingSuiteWriterFrontEnd::InputPtrT input) {
  return (saveRGB || saveRawCloud || input->landmark3d->points.size() > 0);
}

void FittingSuiteWriterFrontEnd::_process(
    FittingSuiteWriterFrontEnd::InputPtrT input) {
  // point order is preserved from 0 to 48(see LandMarkMerger::merge)
  // So it is safe to just put every points in order
  std::stringstream sstream;
  for (const auto &p : input->landmark3d->points) {
    sstream << p.x << ",";
    sstream << p.y << ",";
    sstream << p.z << "\n";
  }

  auto pathPrefix = folder / ("frame_" + std::to_string(frameCount));
  std::ofstream outCsv;
  outCsv.open(pathPrefix.replace_extension(std::string(".csv")));
  outCsv << sstream.str();
  outCsv.close();
  if (saveRGB) {
    telef::io::saveBMPFile(pathPrefix.replace_extension(std::string(".bmp")),
                           *input->rawImage);
  }
  if (saveRawCloud) {
    pcl::io::savePLYFile(pathPrefix.replace_extension(std::string(".ply")),
                         *input->rawCloud);
  }

  frameCount++;
  std::cout << "Captured" << std::endl;
}

FittingSuiteWriterFrontEnd::FittingSuiteWriterFrontEnd(bool ignoreIncomplete,
                                                       bool saveRGB,
                                                       bool saveRawCloud,
                                                       int expectedPointsCount)
    : ignoreIncomplete(ignoreIncomplete), saveRGB(saveRGB),
      saveRawCloud(saveRawCloud), expectedPointsCount(expectedPointsCount),
      frameCount(0) {
  std::time_t t = std::time(nullptr);
  auto localTime = std::localtime(&t);
  std::stringstream folderNameStream;
  folderNameStream << "capture_" << localTime->tm_mday << "-"
                   << localTime->tm_mon + 1 << "-" << localTime->tm_year + 1900
                   << "-" << localTime->tm_hour << "-" << localTime->tm_min
                   << "-" << localTime->tm_sec;
  folder = folderNameStream.str();
  std::experimental::filesystem::create_directory(folder);
}

RecordFakeFrameFrontEnd::RecordFakeFrameFrontEnd(fs::path recordRoot)
    : recordRoot(recordRoot), frameCount(1) {}

void RecordFakeFrameFrontEnd::_process(FrontEnd::InputPtrT input) {
  fs::path prefix = recordRoot / fs::path(std::to_string(frameCount));
  input->save(prefix);
  frameCount++;
}
} // namespace telef::io