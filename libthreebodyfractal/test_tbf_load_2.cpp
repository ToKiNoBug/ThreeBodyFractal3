#include <fractal_utils/core_utils.h>

#include "libthreebodyfractal.h"

int main() {
  using namespace libthreebody;
  using namespace fractal_utils;

  bool ok;
  input_t center_input;
  compute_options opt;
  center_wind<double> wind;

  size_t rows, cols;

  binfile file;
  ok = file.parse_from_file("test.tbf");
  if (!ok) {
    return 1;
  }

  ok = fractal_bin_file_get_information(file, &rows, &cols, &center_input,
                                        &wind, &opt);
  if (!ok) {
    return 1;
  }

  printf("rows = %zu, cols=%zu.\n", rows, cols);

  fractal_map result = fractal_map::create(rows, cols, sizeof(result_t));

  const size_t buffer_bytes = result.byte_count() * 2.5;

  void* buffer = aligned_alloc(32, buffer_bytes);

  ok = fractal_bin_file_get_result(file, &result, buffer, buffer_bytes);
  if (!ok) {
    return 1;
  }

  ok = save_fractal_bin_file("test_rewrite.tbf", center_input, wind, opt,
                             result, buffer, buffer_bytes);
  if (!ok) {
    return 1;
  }

  free(buffer);
  printf("success\n");
  return 0;
}