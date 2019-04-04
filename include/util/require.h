#pragma once

#include <stdexcept>

#define TELEF_REQUIRE(cond) (_telef_require(cond, #cond))

inline void _telef_require(bool cond, const char *expr) {
  if (!cond) {
    std::string reqstr(expr);
    throw std::runtime_error("Requirement is not met: " + reqstr);
  }
}
