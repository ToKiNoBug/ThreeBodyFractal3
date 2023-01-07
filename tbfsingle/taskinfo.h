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
};

struct render_input {
  std::string tbf_file;
};

#endif  // THREEBODYFRACTAL3_TBFSINGLE_TASKINFO_H