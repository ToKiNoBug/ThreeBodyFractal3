#include <fractal_utils/core_utils.h>
#include <fractal_utils/png_utils.h>
#include <omp.h>

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

  input.beg_state.position = {{-1.03584, -0.0215062, 2.08068},
                              {-6.64071, 1.34016, -9.49566},
                              {-6.73013, 8.17534, 1.4536}};
  input.beg_state.position *= rs;

  input.beg_state.velocity = {{0.384347, 0.0969975, -0.50161},
                              {-0.697374, -0.766521, 0.250808},
                              {-0.394691, -0.192819, 0.747116}};
  input.beg_state.velocity *= vs;

  input.mass = {3.20948, 1.84713, 4.6762};
  input.mass *= Ms;

  compute_options opt;
  opt.max_relative_error = 1e-4;
  opt.step_guess = 1e-2 * year;
  opt.time_end = 10 * year;

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

  color_u8c3_many((float*)buffer.data, color_series::jet,
                  buffer.element_count(), (pixel_RGB*)img_u8c3.data);

  const bool ok = write_png("test.png", color_space::u8c3, img_u8c3);

  if (!ok) {
    printf("Failed to write image.\n");
    return 1;
  }

  return 0;
}