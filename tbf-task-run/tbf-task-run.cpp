#include <CLI11.hpp>
#include <tbf-task.h>

#include <iostream>

#include <thread>

#include <omp.h>

#include <filesystem>
#include <libthreebodyfractal.h>

using std::cout, std::endl;

int main(int argc, char **argv) {
  CLI::App app;
  std::string taskfile;

  int max_frames_this_run;

  app.add_option("taskfile", taskfile, "Json task file to run.")
      ->default_val("task.json")
      ->check(CLI::ExistingFile);

  app.add_option("--max-frames", max_frames_this_run,
                 "Maximum frames to compute in this run")
      ->default_val(INT32_MAX)
      ->check(CLI::PositiveNumber);

  CLI11_PARSE(app, argc, argv);

  task_input task;

  if (!load_task_from_json(&task, taskfile)) {
    cout << "Failed to load " << taskfile << endl;
    return 1;
  }

  size_t rows, cols;
  libthreebody::input_t center_input;
  libthreebody::compute_options opt;
  fractal_utils::center_wind<double> original_wind;

  // load center_source
  {
    std::filesystem::path p(task.center_source);
    bool parsed = false;
    if (p.extension() == ".tbf") {
      fractal_utils::binfile file;
      if (!file.parse_from_file(task.center_source.data()) ||
          !libthreebody::fractal_bin_file_get_information(
              file, &rows, &cols, &center_input, &original_wind, &opt)) {
        cout << "Failed to parse " << task.center_source << endl;
        return 1;
      };

      parsed = true;
    }

    if (p.extension() == ".nbt") {
      if (!libthreebody::load_fractal_basical_information_nbt(
              task.center_source, &rows, &cols, &center_input, &original_wind,
              &opt)) {

        cout << "Failed to parse " << task.center_source << endl;
        return 1;
      }
      parsed = true;
    }

    if (!parsed) {
      cout << "Unsupported extension for center source." << endl;
      return 1;
    }
  }

  libthreebody::gpu_mem_allocator gpu_alloc(task.gpu_threads, cols);

  int unfinished = 0;

  for (int frameidx = 0; frameidx < task.frame_count; frameidx++) {

    const std::string filename = task.tbf_filename(frameidx);

    if (!std::filesystem::exists(filename))
      unfinished++;
  }

  if (unfinished <= 0) {
    cout << "All tasks finished." << endl;
    return 0;
  }

  // max_frames_this_run = std::min(max_frames_this_run, unfinished);

  omp_set_num_threads(task.cpu_threads + task.gpu_threads);

  int task_counter = 0;

  fractal_utils::fractal_map map_result(rows, cols,
                                        sizeof(libthreebody::result_t));

  const size_t buffer_cap = map_result.byte_count() * 2.5;

  void *const buffer = fractal_utils::allocate_memory_aligned(32, buffer_cap);

  cout << endl;

  for (int frameidx = 0; frameidx < task.frame_count; frameidx++) {

    const std::string filename = task.tbf_filename(frameidx);

    if (std::filesystem::exists(filename)) {
      break;
    }

    if (task_counter >= max_frames_this_run) {
      cout << task_counter
           << " task(s) is finished, which meets the requirements of "
              "--max-frames, exit."
           << endl;
      return 0;
    }

    task_counter++;
    cout << "\r[ " << task_counter << " / " << unfinished << " : "
         << float(task_counter * 100) / unfinished << "% ] : " << filename
         << endl;

    fractal_utils::center_wind<double> wind = original_wind;
    wind.x_span /= std::pow(task.zoom_speed, frameidx);
    wind.y_span /= std::pow(task.zoom_speed, frameidx);

    libthreebody::compute_frame_cpu_and_gpu(
        center_input, wind, opt, &map_result, &gpu_alloc, task.verbose);
    cout << "Computation finished. Exporting...";
    if (!libthreebody::save_fractal_bin_file(filename, center_input, wind, opt,
                                             map_result, buffer, buffer_cap)) {
      cout << "Failed to export " << filename << endl;
      free(buffer);
      return 1;
    }

    // cout << "frame " << frameidx << ", filename = " << '\n';
  }

  cout << endl;

  free(buffer);

  cout << "All tasks finished." << endl;
  return 0;
}