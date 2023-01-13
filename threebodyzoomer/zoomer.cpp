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
  void *buffer_export;
};

void compute_fun(const fractal_utils::wind_base &, void *custom_ptr,
                 fractal_utils::fractal_map *map_fractal);

void render(const fractal_utils::fractal_map &map_fractal,
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

  mainwindow w(double(1), nullptr, map_size, sizeof(result_t), 4);

  {
    center_wind<double> wind;
    wind.center = {0, 0};
    wind.y_span = 2;
    wind.x_span = wind.y_span / map_size[0] * map_size[1];
    w.set_window(wind);
  }

  custom_parameters custp{gpu_mem_allocator(2, cols), input_t(),
                          compute_options(), nullptr};

  printf("cutp.alloc.size() = %i\n", custp.alloc.size());

  custp.buffer_export = fractal_utils::allocate_memory_aligned(
      32, sizeof(result_t) * rows * cols * 2.5);

  omp_set_num_teams(std::thread::hardware_concurrency() + custp.alloc.size());

  custp.center_input = get_center_input(argc, argV);

  custp.opt.max_relative_error = 1e-4;
  custp.opt.step_guess = 1e-2 * year;
  custp.opt.time_end = 5 * year;

  w.custom_parameters = &custp;
  w.frame_file_extension_list = ".tbf";
  // w.map_fractal = fractal_map::create(rows, cols, sizeof(result_t));
  w.callback_compute_fun = compute_fun;
  w.callback_render_fun = render;
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
    ok = load_parameters_from_D3B3(argv[3], &input.mass, &input.beg_state);
  }

  if (is_tbf) {
    fractal_utils::binfile binfile;
    binfile.parse_from_file(argv[3]);

    ok = libthreebody::fractal_bin_file_get_information(binfile, nullptr,
                                                        nullptr, &input);
  }

  if (!ok) {
    printf("\nError : failed to load parameters from file %s.\n", argv[3]);
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

void render(const fractal_utils::fractal_map &map_fractal,
            const fractal_utils::wind_base &__wind, void *custom_ptr,
            fractal_utils::fractal_map *map_u8c3) {
  using namespace libthreebody;
  /*
    const fractal_utils::center_wind<double> &wind =
        dynamic_cast<const fractal_utils::center_wind<double> &>(__wind);

        */
  custom_parameters *const params =
      reinterpret_cast<custom_parameters *>(custom_ptr);

  render_universial(map_fractal, {0, 0}, params->buffer_export,
                    map_fractal.byte_count() * 2.5, map_u8c3,
                    params->opt.time_end);
}