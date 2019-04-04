#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <tuple>

#include "io/obj.h"
#include "util/require.h"

namespace {
namespace fs = std::experimental::filesystem;
const std::regex re_vline(
    "v\\s+(-*\\d+\\.*\\d*)\\s+(-*\\d+\\.*\\d*)\\s+(-*\\d+\\.*\\d*)[\n\r]*");
const std::regex
    re_fline_quad("f\\s+(\\d+).*\\s+(\\d+).*\\s+(\\d+).*\\s+(\\d+).*[\n\r]*");
const std::regex re_fline_tri("f\\s+(\\d+).*\\s+(\\d+).*\\s+(\\d+).*[\n\r]*");

template <typename T>
void eigen_fill_from_regex(
    Eigen::MatrixBase<T> &mat,
    std::sregex_iterator it,
    bool is_float,
    int num_submatch) {
  std::sregex_iterator end;
  int cnt = 0;
  for (auto i = it; i != end; i++) {
    auto m = *i;
    for (int j = 0; j < num_submatch; j++) {
      mat(cnt, j) = is_float ? std::stof(m[j + 1]) : (std::stoi(m[j + 1]) - 1);
    }
    cnt++;
  }
}
} // namespace

namespace telef::io {
std::pair<Eigen::MatrixXf, Eigen::MatrixXi> readObj(const fs::path &obj_path) {
  std::cout << "OBJPATH" << std::endl;
  std::cout << obj_path << std::endl;
  TELEF_REQUIRE(fs::exists(obj_path));
  std::ifstream fin(obj_path.string());
  // Read all cahracters from the stream
  std::string txt(std::istreambuf_iterator<char>(fin), {});

  auto vmatch_begin = std::sregex_iterator(txt.begin(), txt.end(), re_vline);
  auto fmatch_quad_begin =
      std::sregex_iterator(txt.begin(), txt.end(), re_fline_quad);
  auto fmatch_tri_begin =
      std::sregex_iterator(txt.begin(), txt.end(), re_fline_tri);
  auto end = std::sregex_iterator();

  // Read vertex positions
  Eigen::MatrixXf V(std::distance(vmatch_begin, end), 3);
  eigen_fill_from_regex(V, vmatch_begin, true, 3);

  // Determine if face is triangle or quad
  auto is_quad = 0 != std::distance(fmatch_quad_begin, end);
  auto fit = is_quad ? fmatch_quad_begin : fmatch_tri_begin; // face iterator
  auto fsize = std::distance(fit, end); // face size (count)
  auto fdim = is_quad ? 4 : 3;
  Eigen::MatrixXi F(fsize, fdim);

  // Read face vertex indices
  eigen_fill_from_regex(F, fit, false, fdim);

  return std::make_pair(V, F);
}
} // namespace telef::io
