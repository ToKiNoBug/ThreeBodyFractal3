#include <fractal_utils/png_utils.h>

#include "taskinfo.h"

bool run_render(const render_input &ri) noexcept {
  using namespace fractal_utils;
  using namespace libthreebody;
  color_map_all cm;

  // printf("ri.json_file.empty() = %i\n", ri.json_file.empty());

  if (!ri.json_file.empty()) {
    if (!load_color_map_all_from_file(ri.json_file.c_str(), &cm)) {
      return false;
    }
  } else {
    cm = default_color_map_0;
  }

  binfile binfile;
  bool ok = true;
  if (!binfile.parse_from_file(ri.tbf_file.c_str())) {
    printf("\nError : failed to parse .tbf file.\n");
    return false;
  }

  size_t rows = 0, cols = 0;
  compute_options opt;

  if (!fractal_bin_file_get_information(binfile, &rows, &cols, nullptr, nullptr,
                                        &opt)) {
    printf("\nError : failed to get information.\n");
    return false;
  }

  if (rows <= 0 || cols <= 0) {
    printf("\nError : invalid size : rows = %zu, cols = %zu\n", rows, cols);
    return false;
  }

  fractal_map map_result = fractal_map::create(rows, cols, sizeof(result_t));

  const size_t buffer_cap = map_result.byte_count() * 2.5;
  void *buffer = aligned_alloc(32, buffer_cap);

  fractal_map img_u8c3 = fractal_map::create(rows, cols, 3);

  if (!fractal_bin_file_get_result(binfile, &map_result, buffer, buffer_cap)) {
    free(buffer);
    printf("\nError : failed to get result_t.\n");
    return false;
  }

  color_by_all(map_result.address<result_t>(0), (float *)buffer,
               img_u8c3.address<pixel_RGB>(0), map_result.element_count(),
               opt.time_end, cm);

  if (!write_png(ri.png_file.c_str(), color_space::u8c3, img_u8c3)) {
    free(buffer);
    printf("\nError : failed to export image %s.\n", ri.png_file.c_str());
    return false;
  }
  free(buffer);

  printf("Finished.\n");
  return true;
}