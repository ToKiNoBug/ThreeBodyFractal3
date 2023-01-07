#include <CLI11.hpp>

#include "taskinfo.h"

int main(int argc, char** argv) {
  CLI::App app{"Execute single task of threebodyfractal", "tbfsingle"};

  CLI::App* const compute = app.add_subcommand(
      "compute", "Compute threebody fractal and generate a .tbf file.");
  CLI::App* const render =
      app.add_subcommand("render", "Render a png with given .tbf file.");
  app.require_subcommand(1);

  compute_input ci;

  compute
      ->add_option("--center-input", ci.beg_statue_file,
                   "Get center beginning statue from *.tbf or *.paraD3B3 file")
      ->check(CLI::ExistingDirectory)
      ->required();

  compute->add_option("-o", ci.out_put_file, "Name of generated .tbf file.")
      ->required();

  compute->add_option("--max-time-year", ci.opt.time_end,
                      "Max simulation of year (default=5 year)");
  compute->add_option("--relative-error", ci.opt.max_relative_error,
                      "Max relative error (default=1e-4)");
  compute->add_option("--step-guess-year", ci.opt.step_guess,
                      "Initial step of year (default=0.01 year)");
  compute->add_option("--rows", ci.rows, "Fractal rows")
      ->required()
      ->check(CLI::PositiveNumber);
  compute->add_option("--cols", ci.cols, "Fractal cols")
      ->required()
      ->check(CLI::PositiveNumber);
  compute->add_option("--cpu-threads", ci.cpu_threads, "CPU threads to use")
      ->required()
      ->check(CLI::PositiveNumber);
  compute->add_option("--gpu-threads", ci.gpu_threads, "GPU tasks at one time")
      ->required()
      ->check(CLI::NonNegativeNumber);

  CLI11_PARSE(app, argc, argv);

  if (compute->parsed()) {
    ci.opt.step_guess *= libthreebody::year;
    ci.opt.time_end *= libthreebody::year;
  }

  if (render->parsed()) {
  }

  return 0;
}