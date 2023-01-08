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
      ->check(CLI::ExistingFile)
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
  compute->add_option("--yspan", ci.y_span, "Y span")
      ->required()
      ->check(CLI::PositiveNumber);
  /*
compute->add_option("--xspan", ci.x_span, "X span (default=Y span *cols/rows")
  ->check(CLI::PositiveNumber)
  ->default_val(-1.0);
  */
  compute->add_option("--center-hex", ci.center_hex, "32 digit hex value.")
      ->default_val("0x00000000000000000000000000000000");

  CLI11_PARSE(app, argc, argv);

  if (compute->parsed()) {
    ci.opt.step_guess *= libthreebody::year;
    ci.opt.time_end *= libthreebody::year;

    ci.x_span = (ci.y_span * ci.cols) / ci.rows;

    if (ci.center_hex.length() != 32 && ci.center_hex.length() != 34) {
      printf(
          "Invalid length for center_hex. Expected 32 of 34(with 0x prefix) "
          "but acually %zu\n",
          ci.center_hex.length());
      return 1;
    }

    if (!run_compute(ci)) {
      printf("\nFailed to compute.\n");
      return 1;
    }
  }

  if (render->parsed()) {
  }

  return 0;
}