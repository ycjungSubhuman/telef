#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace telef::io {
void writeLmk(const char *filename, const std::vector<int> &lmk);
void readLmk(const char *filename, std::vector<int> &lmk);
} // namespace telef::io