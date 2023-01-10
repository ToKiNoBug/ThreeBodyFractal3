#ifndef THREEBODYFRACTAL3_TBFSINGLE_TASKINFO_H
#define THREEBODYFRACTAL3_TBFSINGLE_TASKINFO_H

#include "libthreebodyfractal.h"

struct compute_input {
  std::string beg_statue_file;
  libthreebody::compute_options opt{5, 1e-4, 1e-2};
  std::string out_put_file;
  int rows;
  int cols;
  int cpu_threads;
  int gpu_threads;

  double y_span;
  double x_span{-1};
  std::string center_hex;
};

struct render_input {
  std::string tbf_file;
  std::string png_file;
  std::string json_file;
};

bool run_compute(const compute_input &ci) noexcept;

bool run_render(const render_input &ri) noexcept;

#endif  // THREEBODYFRACTAL3_TBFSINGLE_TASKINFO_H