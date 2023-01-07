#include <fractal_utils/core_utils.h>
#include <omp.h>
#include <stdlib.h>

#include "libthreebodyfractal.h"

int main() {
  // load

  using namespace libthreebody;
  using namespace fractal_utils;

  fractal_utils::binfile binfile;
  binfile.parse_from_file("test.tbf");

  {
    int idx = 0;
    for (const auto &blk : binfile.blocks) {
      printf("block %i, tag=%zi, bytes=%zi, offset=%zu.\n", idx, blk.tag,
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

  printf("rows = %zu, cols=%zu.\n", rows, cols);

  printf("information loaded successfully.\n");

  fractal_map map_end_state = fractal_map::create(rows, cols, sizeof(state_t));

  ok = fractal_bin_file_get_end_state(binfile, &map_end_state, true);
  if (!ok) {
    printf("failed to get end_state.\n");
    return 1;
  }

  fractal_map map_end_energy = fractal_map::create(rows, cols, sizeof(double));

  ok = fractal_bin_file_get_end_energy(binfile, &map_end_energy, true);
  if (!ok) {
    printf("failed to get end_energy.\n");
    return 1;
  }

  fractal_map map_end_time = fractal_map::create(rows, cols, sizeof(double));

  ok = fractal_bin_file_get_collide_time(binfile, &map_end_time, true);
  if (!ok) {
    printf("failed to get collide_time.\n");
    return 1;
  }

  fractal_map map_iterate_times = fractal_map::create(rows, cols, sizeof(int));

  ok = fractal_bin_file_get_iterate_time(binfile, &map_iterate_times, true);
  if (!ok) {
    printf("failed to get iterate_time.\n");
    return 1;
  }

  fractal_map map_iterate_fail_times =
      fractal_map::create(rows, cols, sizeof(int));

  ok = fractal_bin_file_get_iterate_fail_time(binfile, &map_iterate_fail_times,
                                              true);
  if (!ok) {
    printf("failed to get iterate_fail_time.\n");
    return 1;
  }

  printf("success.\n");
  return 0;
}