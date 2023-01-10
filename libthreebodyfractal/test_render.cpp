#include <fractal_utils/png_utils.h>

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

  ok = fractal_bin_file_get_result(binfile, &map_result, buffer,
                                   map_result.byte_count() * 2.5);
  if (!ok) {
    return 1;
  }

  fractal_utils::fractal_map img =
      fractal_utils::fractal_map::create(rows, cols, 3);

  color_by_all((const result_t*)map_result.data, (float*)buffer,
               (fractal_utils::pixel_RGB*)img.data, img.element_count(),
               opt.time_end);
  ok = fractal_utils::write_png("./test_all.png",
                                fractal_utils::color_space::u8c3, img);
  if (!ok) {
    return 1;
  }

  color_by_collide_u8c3((const result_t*)map_result.data,
                        (fractal_utils::pixel_RGB*)img.data,
                        img.element_count(), opt.time_end);
  ok = fractal_utils::write_png("./test_collide.png",
                                fractal_utils::color_space::u8c3, img);
  if (!ok) {
    return 1;
  }

  color_by_end_age_u8c3((const result_t*)map_result.data, (float*)buffer,
                        (fractal_utils::pixel_RGB*)img.data,
                        img.element_count(), opt.time_end);
  ok = fractal_utils::write_png("./test_age.png",
                                fractal_utils::color_space::u8c3, img);
  if (!ok) {
    return 1;
  }

  color_by_end_distance_u8c3((const result_t*)map_result.data,
                             (fractal_utils::pixel_RGB*)img.data,
                             img.element_count());
  ok = fractal_utils::write_png("./test_distance.png",
                                fractal_utils::color_space::u8c3, img);
  if (!ok) {
    return 1;
  }
  /*
  color_by_end_distance_and_age_u8c3(
      (const result_t*)map_result.data, (float*)buffer,
      (fractal_utils::pixel_RGB*)img.data, img.element_count(), opt.time_end);
  ok = fractal_utils::write_png("./test_distance_age.png",
                                fractal_utils::color_space::u8c3, img);
  if (!ok) {
    return 1;
  }
  */

  color_by_triangle((const result_t*)map_result.data, (float*)buffer,
                    (fractal_utils::pixel_RGB*)img.data, img.element_count());
  ok = fractal_utils::write_png("./test_triangle.png",
                                fractal_utils::color_space::u8c3, img);
  if (!ok) {
    return 1;
  }

  printf("success\n");

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