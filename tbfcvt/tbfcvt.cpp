#include <CLI11.hpp>
#include <iostream>
#include <libthreebodyfractal.h>

bool execute(std::string_view src, std::string_view dst, size_t __rows,
             size_t __cols,
             const fractal_utils::center_wind<double> &__wind) noexcept;

int main(int argc, char **argv) {

  CLI::App app;

  std::string source_file;
  std::string dest_file;
  fractal_utils::center_wind<double> __wind;

  size_t rows, cols;
  app.add_option("source file", source_file, "Source file")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option("-o", dest_file, "File to generate")->required();

  app.add_option("--rows", rows, "Rows for .paraD3B3 file.")->default_val(0);
  app.add_option("--cols", cols, "Cols for .paraD3B3 file.")->default_val(0);

  app.add_option("--center-x", __wind.center[0], "X position of center")
      ->default_val(std::nan(nullptr));
  app.add_option("--center-y", __wind.center[1], "Y position of center")
      ->default_val(std::nan(nullptr));

  app.add_option("--span-x", __wind.x_span, "X range of window")
      ->default_val(std::nan(nullptr));
  app.add_option("--span-y", __wind.y_span, "Y range of window")
      ->default_val(std::nan(nullptr));

  CLI11_PARSE(app, argc, argv);

  if (!execute(source_file, dest_file, rows, cols, __wind)) {
    std::cout << "Failed." << std::endl;
    return 1;
  }

  std::cout << "success" << std::endl;

  return 0;
}

bool execute(std::string_view src, std::string_view dst, size_t __rows,
             size_t __cols,
             const fractal_utils::center_wind<double> &__wind) noexcept {
  const std::filesystem::path src_path = src.data(), dst_path = dst.data();

  bool is_src_parsed = false;

  if (src_path.extension() == dst_path.extension()) {
    std::error_code ec;
    std::filesystem::copy(src_path, dst_path, ec);

    if (ec) {
      std::cout << "Failed to copy. Error code = " << ec << std::endl;
      return false;
    }

    return true;
  }

  libthreebody::input_t center_input;
  libthreebody::compute_options opt;
  size_t rows, cols;
  fractal_utils::center_wind<double> wind;

  if (src_path.extension() == ".json") {
    if (!libthreebody::load_fractal_basical_information_json(
            src, &rows, &cols, &center_input, &wind, &opt)) {
      std::cout << "Failed to parse " << src << std::endl;
      return false;
    }
    is_src_parsed = true;
  }

  if (src_path.extension() == ".tbf") {

    fractal_utils::binfile file;
    if (!file.parse_from_file(src.data())) {
      std::cout << "Failed to parse " << src << std::endl;
      return false;
    }

    if (!libthreebody::fractal_bin_file_get_information(
            file, &rows, &cols, &center_input, &wind, &opt)) {
      std::cout << "Failed to parse " << src << std ::endl;
      return false;
    }

    is_src_parsed = true;
  }

  if (src_path.extension() == ".nbt") {
    if (!libthreebody::load_fractal_basical_information_nbt(
            src, &rows, &cols, &center_input, &wind, &opt)) {
      std::cout << "Failed to parse " << src << std ::endl;
      return false;
    }
    is_src_parsed = true;
  }

  if (src_path.extension() == ".paraD3B3") {
    if (__rows <= 0 || __cols <= 0) {
      std::cout << "Invalid size : rows = " << __rows << ", cols = " << __cols
                << std::endl;
      return false;
    }

    rows = __rows;
    cols = __cols;

    if (std::isnan(__wind.center[0]) || std::isnan(__wind.center[1])) {

      std::cout << "Invalid center : [" << __wind.center[0] << ", "
                << __wind.center[1] << "]" << std::endl;
      return false;
    }
    if (std::isnan(__wind.x_span) || std::isnan(__wind.y_span)) {

      std::cout << "Invalid span : x_span = " << __wind.x_span
                << ", y_span = " << __wind.y_span << std::endl;
      return false;
    }

    wind = __wind;

    if (!libthreebody::load_parameters_from_D3B3(
            src, &center_input.mass, &center_input.beg_state, &opt)) {
      std::cout << "Failed to parse " << src << std ::endl;
      return false;
    }
    is_src_parsed = true;
  }

  if (!is_src_parsed) {
    std::cout << "Error : unknown source file extension" << std::endl;
    return false;
  }

  bool is_dest_written = false;

  if (dst_path.extension() == ".tbf" || dst_path.extension() == ".nbt") {
    if (!libthreebody::save_fractal_basical_information_binary(
            dst, center_input, wind, opt, {rows, cols})) {
      std::cout << "Failed to generate " << dst << std::endl;
      return false;
    }

    is_dest_written = true;
  }

  if (dst_path.extension() == ".json") {
    if (!libthreebody::save_fractal_basical_information_json(
            dst, center_input, wind, opt, {rows, cols})) {
      std::cout << "Failed to generate " << dst << std::endl;
      return false;
    }

    is_dest_written = true;
  }

  if (dst_path.extension() == ".paraD3B3") {
    std::cout << "Exporting as .paraD3B3 is not allowed." << std::endl;
    return false;
  }

  if (!is_dest_written) {
    std::cout << "Error : unknown dest file extension" << std::endl;
    return false;
  }

  return true;
}