#include <fstream>
#include <string>
#include <vector>

#include "io/landmark.h"

namespace telef::io {
void writeLmk(const char *filename, const std::vector<int> &lmk) {
  std::ofstream f(filename);
  f << lmk.size() << "\n\n";
  for (const auto &l : lmk) {
    f << l << "\n";
  }
  f.close();
}

void readLmk(const char *filename, std::vector<int> &lmk) {
  std::ifstream f(filename);
  std::string c;
  f >> c;
  int count = std::stoi(c);
  std::string pt;
  while (f >> pt) {
    lmk.push_back(std::stoi(pt));
  }
}
} // namespace telef::io
