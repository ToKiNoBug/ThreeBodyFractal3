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
  void *buffer_export;
};

void compute_fun(const fractal_utils::wind_base &, void *custom_ptr,
                 fractal_utils::fractal_map *map_fractal);

void render_fun_end_age_only(
    const fractal_utils::fractal_map &map_fractal,
    const fractal_utils::wind_base &window, void *custom_ptr,
    fractal_utils::fractal_map *map_u8c3_do_not_resize);

void render_fun_end_state_only(
    const fractal_utils::fractal_map &map_fractal,
    const fractal_utils::wind_base &window, void *custom_ptr,
    fractal_utils::fractal_map *map_u8c3_do_not_resize);

void render_fun_all(const fractal_utils::fractal_map &map_fractal,
                    const fractal_utils::wind_base &window, void *custom_ptr,
                    fractal_utils::fractal_map *map_u8c3_do_not_resize);

bool export_bin_file(const fractal_utils::fractal_map &map_fractal,
                     const fractal_utils::wind_base &window, void *custom_ptr,
                     const fractal_utils::fractal_map &map_u8c3_do_not_resize,
                     const char *filename);

int main(int argc, char **argV) {
  QApplication app(argc, argV);

  const std::array<int, 2> map_size = get_map_size(argc, argV);
  const int rows = map_size[0];
  const int cols = map_size[1];

  using namespace fractal_utils;
  using namespace libthreebody;

  mainwindow w(double(1), nullptr, map_size, 4);

  {
    center_wind<double> wind;
    wind.center = {0, 0};
    wind.y_span = 2;
    wind.x_span = wind.y_span / map_size[0] * map_size[1];
    w.set_window(wind);
  }

  custom_parameters custp{
      gpu_mem_allocator(2, cols), input_t(), compute_options(),
      fractal_map::create(rows, cols, sizeof(float)), nullptr};

  printf("size of cutp.alloc.size = %i\n", custp.alloc.size());
  {
#ifdef WIN32
    void *const buffer =
        _aligned_malloc(sizeof(result_t) * rows * cols * 2.5, 32);
#else
    void *const buffer =
        aligned_alloc(32, sizeof(result_t) * rows * cols * 2.5);
#endif

    custp.buffer_export = buffer;
  }

  omp_set_num_teams(std::thread::hardware_concurrency() + custp.alloc.size());

  custp.center_input = get_center_input(argc, argV);

  custp.opt.max_relative_error = 1e-4;
  custp.opt.step_guess = 1e-2 * year;
  custp.opt.time_end = 5 * year;

  w.custom_parameters = &custp;
  w.frame_file_extension_list = ".tbf";
  w.map_fractal = fractal_map::create(rows, cols, sizeof(result_t));
  w.callback_compute_fun = compute_fun;
  w.callback_render_fun = render_fun_all;
  w.callback_export_fun = export_bin_file;

  w.show();

  w.display_range();
  // w.compute_and_paint();
  int ret = app.exec();

  free(custp.buffer_export);

  return ret;
}

std::array<int, 2> get_map_size(int argc,
                                const char *const *const argv) noexcept {
  std::array<int, 2> ret{160, 160};

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

  const bool is_paraD3B3 = (src_filename.extension() == ".paraD3B3");
  const bool is_tbf = (src_filename.extension() == ".tbf");

  if (!is_paraD3B3 && !is_tbf) {
    printf("\nError : invalid extension for src_filename.\n");
    exit(1);
    return {};
  }

  bool ok = true;
  if (is_paraD3B3) {
    ok = load_parameters_from_D3B3(src_filename.c_str(), &input.mass,
                                   &input.beg_state);
  }

  if (is_tbf) {
    fractal_utils::binfile binfile;
    binfile.parse_from_file(src_filename.c_str());

    ok = libthreebody::fractal_bin_file_get_information(binfile, nullptr,
                                                        nullptr, &input);
  }

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
  double wtime = omp_get_wtime();
  libthreebody::compute_frame_cpu_and_gpu(
      params->center_input, wind, params->opt, map_fractal, &params->alloc);
  wtime = omp_get_wtime() - wtime;

  printf("%zu computations finished in %F seconds. %F ms per simulation.\n",
         map_fractal->element_count(), wtime,
         1000 * wtime / map_fractal->element_count());
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
          1.0f - float(map_fractal.at<result_t>(r, c).end_time) / max_time;
    }

    fractal_utils::color_u8c3_many(
        &params->buffer_float32.at<float>(r, 0),
        fractal_utils::color_series::parula, map_fractal.cols,
        &map_u8c3->at<fractal_utils::pixel_RGB>(r, 0));
  }
}

void render_fun_end_state_only(const fractal_utils::fractal_map &map_fractal,
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
    color_by_collide_u8c3(map_fractal.address<result_t>(r, 0),
                          map_u8c3->address<fractal_utils::pixel_RGB>(r, 0),
                          map_fractal.cols, params->opt.time_end);
  }
}

void render_fun_all(const fractal_utils::fractal_map &map_fractal,
                    const fractal_utils::wind_base &__wind, void *custom_ptr,
                    fractal_utils::fractal_map *map_u8c3) {
  using namespace libthreebody;

  const fractal_utils::center_wind<double> &wind =
      dynamic_cast<const fractal_utils::center_wind<double> &>(__wind);
  custom_parameters *const params =
      reinterpret_cast<custom_parameters *>(custom_ptr);

  const float_t max_time = params->opt.time_end;
#pragma omp parallel for schedule(static)
  for (int r = 0; r < map_fractal.rows; r++) {
    color_by_all(map_fractal.address<result_t>(r, 0),
                 params->buffer_float32.address<float>(r, 0),
                 map_u8c3->address<fractal_utils::pixel_RGB>(r, 0),
                 map_fractal.cols, params->opt.time_end);
  }
}

bool export_bin_file(const fractal_utils::fractal_map &map_fractal,
                     const fractal_utils::wind_base &__wind, void *custom_ptr,
                     const fractal_utils::fractal_map &map_u8c3_do_not_resize,
                     const char *filename) {
  const fractal_utils::center_wind<double> &wind =
      dynamic_cast<const fractal_utils::center_wind<double> &>(__wind);
  custom_parameters *const params =
      reinterpret_cast<custom_parameters *>(custom_ptr);

  return libthreebody::save_fractal_bin_file(
      filename, params->center_input, wind, params->opt, map_fractal,
      params->buffer_export, 2.5 * map_fractal.byte_count());
}
