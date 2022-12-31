#include <fractal_utils/core_utils.h>
#include <fractal_utils/png_utils.h>
#include <omp.h>
#include <stdlib.h>

#include <cstring>
#include <memory>
#include <thread>

#include "threebodyfractal.h"

int main() {
  using namespace libthreebody;
  using namespace fractal_utils;

  const int rows = 320, cols = 320;

  fractal_map map_result =
      fractal_map::create(rows, cols, sizeof(libthreebody::result_t));

  omp_set_num_threads(std::thread::hardware_concurrency());

  center_wind<double> wind;

  wind.center = {0, 0};
  wind.y_span = 2;
  wind.x_span = wind.y_span / map_result.rows * map_result.cols;

  input_t input;
  /*-0.126186392581979
-0.262340533926015
-0.0512572533808924
-0.0609798828559385
0.0762336331145880
0.128100099879297*/
  input.beg_state.position = {
      {-2.27203257188855, 1.09628120907693, 1.17575136281162},
      {-0.519959453298081, -1.98504043661515, 2.50499988991323},
      {0, 0, 0}};
  input.beg_state.position *= rs;

  input.beg_state.velocity = {
      {-0.126186392581979, -0.0512572533808924, 0.0762336331145880},
      {-0.262340533926015, -0.0609798828559385, 0.128100099879297},
      {0, 0, 0}};
  input.beg_state.velocity *= vs;

  input.mass = {1, 2, 3};
  input.mass *= Ms;

  compute_options opt;
  opt.max_relative_error = 1e-4;
  opt.step_guess = 1e-2 * year;
  opt.time_end = 5 * year;

  double wtime;
  wtime = omp_get_wtime();
  compute_frame(input, wind, opt, &map_result);
  wtime = omp_get_wtime() - wtime;

  printf("%i simulations finished in %F seconds. %F ms per simulation.\n",
         rows * cols, wtime, wtime / (rows * cols) * 1000);

  fractal_map img_u8c3 = fractal_map::create(rows, cols, 3);

  memset(img_u8c3.data, 0, img_u8c3.byte_count());

  fractal_map buffer = fractal_map::create(rows, cols, sizeof(float));

  for (int i = 0; i < map_result.element_count(); i++) {
    buffer.at<float>(i) =
        1 - map_result.at<result_t>(i).end_time / opt.time_end;
  }

  color_u8c3_many((float *)buffer.data, color_series::jet,
                  buffer.element_count(), (pixel_RGB *)img_u8c3.data);

  bool ok = write_png("test.png", color_space::u8c3, img_u8c3);

  if (!ok) {
    printf("Failed to write image.\n");
    return 1;
  }
  // save
  {
    const size_t buffer_bytes = map_result.byte_count() * 2.5;
#ifdef WIN32
    void *const buffer = _aligned_malloc(map_result.byte_count() * 2.5, 32);
#else

    void *const buffer = aligned_alloc(32, map_result.byte_count() * 2.5);
#endif

    ok = libthreebody::save_fractal_bin_file("test.tbf", input, wind, opt,
                                             map_result, buffer, buffer_bytes);
    free(buffer);

    if (!ok) {
      printf("Failed to save fractal bin file.\n");
      return 1;
    }
  }

  return 0;
}