#pragma once

#include <cstddef>

namespace telef::util {
template <class InT, class OutT>
void convertArray(const InT *iarr, OutT *oarr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    oarr[i] = static_cast<OutT>(iarr[i]);
  }
}
} // namespace telef::util
