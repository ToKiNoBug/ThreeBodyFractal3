#include <fractal_utils/core_utils.h>
#include <fractal_utils/zoom_utils.h>
#include <libthreebodyfractal.h>
#include <omp.h>
#include <stdio.h>

#include <QApplication>
#include <cstring>
#include <filesystem>
#include <thread>

std::array<int, 2> get_map_size(int argc,
                                const char *const *const argv) noexcept;

libthreebody::input_t get_center_input(int argc,
                                       const char *const *const argv) noexcept;

struct custom_parameters {
  libthreebody::gpu_mem_allocator alloc;
  libthreebody::input_t center_input;
  libthreebody::compute_options opt;
  fractal_utils::fractal_map buffer_float32;
};

void compute_fun(const fractal_utils::wind_base &, void *custom_ptr,
                 fractal_utils::fractal_map *map_fractal);

void render_fun_end_age_only(
    const fractal_utils::fractal_map &map_fractal,
    const fractal_utils::wind_base &window, void *custom_ptr,
    fractal_utils::fractal_map *map_u8c3_do_not_resize);

int main(int argc, char **argV) {
  QApplication app(argc, argV);

  const std::array<int, 2> map_size = get_map_size(argc, argV);
  const int rows = map_size[0];
  const int cols = map_size[1];

  using namespace fractal_utils;
  using namespace libthreebody;

  mainwindow w(double(1), nullptr, map_size);

  {
    center_wind<double> wind;
    wind.center = {0, 0};
    wind.y_span = 2;
    wind.x_span = wind.y_span / map_size[0] * map_size[1];
    w.set_window(wind);
  }

  custom_parameters custp{gpu_mem_allocator(1, cols), input_t(),
                          compute_options(),
                          fractal_map::create(rows, cols, sizeof(float))};

  printf("size of cutp.alloc.size = %i\n", custp.alloc.size());

  omp_set_num_teams(std::thread::hardware_concurrency() + custp.alloc.size());

  custp.center_input = get_center_input(argc, argV);

  custp.opt.max_relative_error = 1e-4;
  custp.opt.step_guess = 1e-2 * year;
  custp.opt.time_end = 5 * year;

  w.custom_parameters = &custp;
  w.frame_file_extension_list = ".tbf";
  w.map_fractal = fractal_map::create(rows, cols, sizeof(result_t));
  w.callback_compute_fun = compute_fun;
  w.callback_render_fun = render_fun_end_age_only;
  w.callback_export_fun;

  w.show();

  w.display_range();
  // w.compute_and_paint();

  return app.exec();
}

std::array<int, 2> get_map_size(int argc,
                                const char *const *const argv) noexcept {
  std::array<int, 2> ret{320, 320};

  if (argc <= 2) {
    return ret;
  }
  int temp;
  temp = std::atoi(argv[1]);

  if (temp <= 0) {
    printf("Invalid rows : %i. Expected positive integer.", temp);
    exit(1);
    return {};
  }

  ret[0] = temp;

  temp = std::atoi(argv[2]);

  if (temp <= 0) {
    printf("Invalid cols : %i. Expected positive integer.", temp);
    exit(1);
    return {};
  }
  ret[1] = temp;
  return ret;
}

libthreebody::input_t get_center_input(int argc,
                                       const char *const *const argv) noexcept {
  using namespace libthreebody;
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

  if (argc < 4) {
    return input;
  }

  std::filesystem::path src_filename = argv[3];

  if (src_filename.extension() != ".paraD3B3") {
    printf("\nError : invalid extension for src_filename.\n");
    exit(1);
    return {};
  }
  bool ok = true;
  ok = load_parameters_from_D3B3(src_filename.c_str(), &input.mass,
                                 &input.beg_state);

  if (!ok) {
    printf("\nError : failed to load parameters from file %s.\n",
           src_filename.c_str());
    exit(1);
    return {};
  }

  return input;
}

void compute_fun(const fractal_utils::wind_base &__wind, void *custom_ptr,
                 fractal_utils::fractal_map *map_fractal) {
  const fractal_utils::center_wind<double> &wind =
      dynamic_cast<const fractal_utils::center_wind<double> &>(__wind);

  custom_parameters *const params =
      reinterpret_cast<custom_parameters *>(custom_ptr);

  libthreebody::compute_frame_cpu_and_gpu(
      params->center_input, wind, params->opt, map_fractal, &params->alloc);
}

void render_fun_end_age_only(const fractal_utils::fractal_map &map_fractal,
                             const fractal_utils::wind_base &__wind,
                             void *custom_ptr,
                             fractal_utils::fractal_map *map_u8c3) {
  using namespace libthreebody;

  const fractal_utils::center_wind<double> &wind =
      dynamic_cast<const fractal_utils::center_wind<double> &>(__wind);
  custom_parameters *const params =
      reinterpret_cast<custom_parameters *>(custom_ptr);

  const float_t max_time = params->opt.time_end;

#pragma omp parallel for schedule(static)
  for (int r = 0; r < map_fractal.rows; r++) {
    for (int c = 0; c < map_fractal.cols; c++) {
      params->buffer_float32.at<float>(r, c) =
          float(map_fractal.at<result_t>(r, c).end_time) / max_time;
    }

    fractal_utils::color_u8c3_many(
        &params->buffer_float32.at<float>(r, 0),
        fractal_utils::color_series::hsv, map_fractal.cols,
        &map_u8c3->at<fractal_utils::pixel_RGB>(r, 0));
  }
}