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
  app.add_option("taskfile", taskfile, "Json task file to run.")
      ->default_val("task.json")
      ->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  task_input task;

  if (!load_task_from_json(&task, taskfile)) {
    cout << "Failed to load " << taskfile << endl;
    return 1;
  }

  size_t rows, cols;
  libthreebody::input_t center_input;
  libthreebody::compute_options opt;
  fractal_utils::center_wind<double> wind;
  {
    std::filesystem::path p(task.center_source);
    bool parsed = false;
    if (p.extension() == ".tbf") {
      fractal_utils::binfile file;
      if (!file.parse_from_file(task.center_source.data()) ||
          !libthreebody::fractal_bin_file_get_information(
              file, &rows, &cols, &center_input, &wind, &opt)) {
        cout << "Failed to parse " << task.center_source << endl;
        return 1;
      };

      parsed = true;
    }

    if (p.extension() == ".nbt") {
      if (!libthreebody::load_fractal_basical_information_nbt(
              task.center_source, &rows, &cols, &center_input, &wind, &opt)) {

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

  return 0;
}