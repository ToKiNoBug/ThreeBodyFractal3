#include <fractal_utils/core_utils.h>
#include <omp.h>
#include <stdlib.h>

#include <cstring>
#include <memory>
#include <thread>

#include "threebodyfractal.h"

int main() {
  // load

  using namespace libthreebody;
  using namespace fractal_utils;

  fractal_utils::binfile binfile;
  binfile.parse_from_file("test.tbf");

  size_t rows, cols;
  input_t center_input;
  center_wind<double> wind;
  compute_options opt;

  bool ok = fractal_bin_file_get_information(binfile, &rows, &cols,
                                             &center_input, &wind, &opt);
  if (!ok) {
    printf("failed to get information.\n");
    return 1;
  }

  printf("information loaded successfully.\n");

  return 0;
}