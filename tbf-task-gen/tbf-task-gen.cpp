#include <CLI11.hpp>
#include <string>
#include <thread>

#include "tbf-task.h"

int main(int argc, char **argv) {
  CLI::App app;
  task_input ti;

  std::string json_file;

  app.add_option("--center-source", ti.center_source,
                 "A .nbt or .tbf file that")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("--zoom-speed", ti.zoom_speed)
      ->default_val(2.0)
      ->check(CLI::PositiveNumber);
  app.add_option("--frame-count", ti.frame_count)
      ->check(CLI::PositiveNumber)
      ->required();

  app.add_option("--fps", ti.fps)->check(CLI::PositiveNumber)->default_val(60);

  app.add_option("--extra-fps", ti.extra_fps)
      ->default_val(0)
      ->check(CLI::NonNegativeNumber);

  app.add_option("-o", json_file, "Generated task file.")
      ->default_val("task.json");
  app.add_option("--tbf-prefix", ti.tbf_file_prefix,
                 "Filename prefix of all tbf files")
      ->default_val("");
  app.add_option("--png-prefix", ti.png_file_prefix,
                 "Filename prefix of all png files")
      ->default_val("");
  app.add_option("--cpu-threads", ti.cpu_threads)
      ->default_val(std::thread::hardware_concurrency())
      ->check(CLI::PositiveNumber);
  app.add_option("--gpu-threads", ti.gpu_threads)
      ->default_val(0)
      ->check(CLI::NonNegativeNumber);

  CLI::Option *const opt_render =
      app.add_option("--render-json", ti.render_json,
                     "Json used to guide rendering.")
          ->check(CLI::ExistingFile);

  app.add_flag("--verbose", ti.verbose,
               "Display the progress of each computation.");

  CLI11_PARSE(app, argc, argv);

  if (opt_render->empty()) {
    ti.render_json = "default";
  }

  if (!save_task_to_json(ti, json_file)) {
    return 1;
  }

  if (!create_directories_by_filename(ti.tbf_filename(0))) {
    std::cout << "Failed to create the parent directory for tbf files"
              << std::endl;
    return 1;
  }

  // std::cout << ti.png_filename(0, 0) << std::endl;

  if (!create_directories_by_filename(ti.png_filename(0, 0))) {
    std::cout << "Failed to create the parent directory for png files"
              << std::endl;
    return 1;
  }

  std::cout << "Task generated." << std::endl;

  return 0;
}
