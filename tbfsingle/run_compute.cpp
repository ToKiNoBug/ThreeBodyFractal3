#include <omp.h>

#include <filesystem>

#include "taskinfo.h"

bool load_beg_status(const std::string& file,
                     libthreebody::input_t* const) noexcept;

bool run_compute(const compute_input& ci) noexcept {
  using namespace libthreebody;
  using namespace fractal_utils;
  gpu_mem_allocator alloc(ci.gpu_threads, ci.rows);
  omp_set_num_threads(ci.cpu_threads + ci.gpu_threads);

  input_t center_input;

  if (!load_beg_status(ci.beg_statue_file, &center_input)) {
    return false;
  }

  fractal_map map_result =
      fractal_map::create(ci.rows, ci.cols, sizeof(result_t));

  center_wind<double> wind;

  wind.x_span = ci.x_span;
  wind.y_span = ci.y_span;
  bool ok = true;
  ok = hex_to_binary(ci.center_hex.data(), wind.center_data());
  if (!ok) {
    printf("\nFailed to get center hex.\n");
    return false;
  }
  double wtime = omp_get_wtime();
  compute_frame_cpu_and_gpu(center_input, wind, ci.opt, &map_result, &alloc);

  wtime = omp_get_wtime() - wtime;
  printf("%i simulations finished in %F seconds. %F ms per simulation.\n",
         ci.rows * ci.cols, wtime, 1000 * wtime / (ci.rows * ci.cols));

  const size_t buffer_bytes = map_result.byte_count() * 2.5;
  void* const buffer = aligned_alloc(32, buffer_bytes);

  ok = save_fractal_bin_file(ci.out_put_file, center_input, wind, ci.opt,
                             map_result, buffer, buffer_bytes);
  if (!ok) {
    free(buffer);
    printf("\nFailed to save tbf file.\n");
    return false;
  }
  free(buffer);

  printf("Fractal bin file generated.\n");
  return true;
}

bool load_beg_status(const std::string& file,
                     libthreebody::input_t* const ret) noexcept {
  std::filesystem::path pth(file);
  bool ok = true;
  if (pth.extension() == ".tbf") {
    fractal_utils::binfile binfile;
    ok = binfile.parse_from_file(file.c_str());
    if (!ok) return false;

    ok = libthreebody::fractal_bin_file_get_information(binfile, nullptr,
                                                        nullptr, ret);
    if (!ok) return false;
    return true;
  }

  if (pth.extension() == ".paraD3B3") {
    ok = libthreebody::load_parameters_from_D3B3(file.data(), &ret->mass,
                                                 &ret->beg_state);
    if (!ok) return false;
    return true;
  }

  printf("\nError : unknown beg status file %s\n", file.c_str());
  return false;
}