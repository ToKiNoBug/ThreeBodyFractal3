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

  {
    int idx = 0;
    for (const auto &blk : binfile.blocks) {
      printf("block %i, tag=%lli, bytes=%llu, offset=%llu.\n", idx, blk.tag,
             blk.bytes, blk.file_offset);
      idx++;
    }
  }

  size_t rows = 0, cols = 0;
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

  fractal_map map = fractal_map::create(rows, cols, sizeof(state_t));

  ok = fractal_bin_file_get_end_state(binfile, &map, true);
  if (!ok) {
    printf("failed to get end_state.\n");
    return 1;
  }

  printf("end state loaded successfully.\n");

  return 0;
}