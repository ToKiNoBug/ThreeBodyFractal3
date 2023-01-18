#include <CLI11.hpp>
#include <nlohmann/json.hpp>
#include <string>

struct task_input {
  size_t rows;
  size_t cols;

  double y_span;

  std::string center_hex;
};

int main(int argc, char **argv) {
  CLI::App app;
  task_input ti;
  app.add_option("--rows", ti.rows, "Fractal rows.")
      ->default_val(320)
      ->check(CLI::PositiveNumber);
  app.add_option("--cols", ti.cols, "Fractal cols.")
      ->default_val(320)
      ->check(CLI::PositiveNumber);

  CLI11_PARSE(app, argc, argv);

  return 0;
}