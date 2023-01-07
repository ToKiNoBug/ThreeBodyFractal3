#include "libthreebodyfractal.h"

std::array<int, 2> get_size(const fractal_utils::binfile&,
                            libthreebody::compute_options* opt) noexcept;

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("\nError : expected filename.\n");
    return 1;
  }

  const char* const tbf_filename = argv[1];

  fractal_utils::binfile binfile;

  bool ok = binfile.parse_from_file(tbf_filename);

  if (!ok) {
    return 1;
  }
  using namespace libthreebody;
  compute_options opt;
  std::array<int, 2> map_size = get_size(binfile, &opt);

  const int rows = map_size[0];
  const int cols = map_size[1];

  fractal_utils::fractal_map map_result =
      fractal_utils::fractal_map::create(rows, cols, sizeof(result_t));

  void* const buffer = aligned_alloc(32, map_result.byte_count() * 2.5);

  free(buffer);
  return 0;
}

std::array<int, 2> get_size(const fractal_utils::binfile& binfile,
                            libthreebody::compute_options* opt) noexcept {
  size_t rows, cols;

  const bool ok = libthreebody::fractal_bin_file_get_information(
      binfile, &rows, &cols, nullptr, nullptr, opt);

  if (!ok) {
    exit(1);
    return {};
  }

  return {int(rows), int(cols)};
}