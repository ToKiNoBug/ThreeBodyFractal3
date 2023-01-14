#include <CLI11.hpp>
#include <iostream>
#include <libthreebodyfractal.h>

bool execute(std::string_view src, std::string_view dst, size_t __rows,
             size_t __cols) noexcept;

int main(int argc, char **argv) {

  CLI::App app;

  std::string source_file;
  std::string dest_file;

  size_t rows, cols;
  app.add_option("source file", source_file, "Source file")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option("-o", dest_file, "File to generate")->required();

  app.add_option("--rows", rows, "Rows for .paraD3B3 file.")->default_val(0);
  app.add_option("--cols", cols, "Cols for .paraD3B3 file.")->default_val(0);

  CLI11_PARSE(app, argc, argv);

  if (!execute(source_file, dest_file, rows, cols)) {
    std::cout << "Failed." << std::endl;
    return 1;
  }

  std::cout << "success" << std::endl;

  return 0;
}

bool execute(std::string_view src, std::string_view dst, size_t __rows,
             size_t __cols) noexcept {
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

  if (src_path.extension() == ".tbf" || src_path.extension() == ".nbt") {
    fractal_utils::binfile file;

    std::vector<uint8_t> buffer;
    if (src_path.extension() == ".tbf") {
      if (!file.parse_from_file(src.data())) {
        std::cout << "Failed to parse " << src << std::endl;
        return false;
      }
    } else {
      const size_t filesize = std::filesystem::file_size(src_path);
      buffer.reserve(filesize);

      std::ifstream ifs(src.data(), std::ios::binary | std::ios::in);
      if (!ifs.is_open()) {
        std::cout << "Failed to open " << src << std::endl;
        return false;
      }

      ifs.read((char *)buffer.data(), filesize);
      const size_t read_bytes = ifs.gcount();

      if (read_bytes != filesize) {
        std::cout << "Failed to read " << src << " : read " << read_bytes
                  << " bytes but the file size is " << filesize << " bytes."
                  << std::endl;
        return false;
      }

      fractal_utils::data_block blk;
      blk.tag = libthreebody::fractal_binfile_tag::basical_information;
      blk.data = buffer.data();
      blk.bytes = filesize;

      file.blocks.emplace_back(blk);
    }

    // here binfile is ready.

    if (!libthreebody::fractal_bin_file_get_information(
            file, &rows, &cols, &center_input, &wind, &opt)) {
      std::cout << "Failed to parse " << src << std ::endl;
      return false;
    }

    // buffer deconstruct here
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