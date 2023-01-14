#include "libthreebodyfractal.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

bool libthreebody::save_fractal_basical_information_json(
    std::string_view filename, const input_t &center_input,
    const fractal_utils::center_wind<double> &wind, const compute_options &opt,
    const std::array<size_t, 2> &size_rc) noexcept {
  if (std::filesystem::path(filename.data()).extension() != ".json") {
    printf("\nError : wrong extension name. Expected .json\n");
    return false;
  }

  using njson = nlohmann::json;

  njson jobj;
  jobj["rows"] = size_rc[0];
  jobj["cols"] = size_rc[1];
  {
    njson::array_t arr_mass(3), arr_beg_state(18);
    for (int i = 0; i < 3; i++) {
      arr_mass[i] = center_input.mass[i] / Ms;
    }

    for (int i = 0; i < 9; i++) {
      arr_beg_state[i] = center_input.beg_state.position(i) / rs;
    }

    for (int i = 0; i < 9; i++) {
      arr_beg_state[i + 9] = center_input.beg_state.velocity(i) / vs;
    }

    jobj["center_input"] = {{"mass", arr_mass},
                            {"initial_state", arr_beg_state}};
  }

  jobj["window"] = {{"center", {{"x", wind.center[0]}, {"y", wind.center[1]}}},
                    {"x_span", wind.x_span},
                    {"y_span", wind.y_span}};

  jobj["compute_option"] = {{"time_end", opt.time_end / year},
                            {"max_relative_error", opt.max_relative_error},
                            {"step_guess", opt.step_guess / year}};

  std::ofstream ofs(filename.data(), std::ios::out);

  if (!ofs) {
    printf("\nError : failed to create or open file %s.\n", filename.data());
    ofs.close();
    return false;
  }

  ofs << jobj;

  ofs.close();

  return true;
}

bool libthreebody::load_fractal_basical_information_json(
    std::string_view filename, size_t *const rows_dest, size_t *const cols_dest,
    input_t *const center_input_dest,
    fractal_utils::center_wind<double> *const wind_dest,
    compute_options *const opt_dest) noexcept {
  if (std::filesystem::path(filename.data()).extension() != ".json") {
    printf("\nError : wrong extension name. Expected .json\n");
    return false;
  }
  using njson = nlohmann::json;

  njson jobj;

  {
    std::ifstream ifs(filename.data(), std::ios::in);
    if (!ifs) {
      printf("\nError : failed to open file %s.\n", filename.data());
      ifs.close();
      return false;
    }

    ifs >> jobj;
    ifs.close();
  }

  if (rows_dest != nullptr) {
    if (!jobj.contains("rows") || !jobj.at("rows").is_number_integer()) {
      printf("\nError : no valid value for rows\n");
      return false;
    }
    *rows_dest = jobj.at("rows");
  }

  if (cols_dest != nullptr) {
    if (!jobj.contains("cols") || !jobj.at("cols").is_number_integer()) {
      printf("\nError : no valid value for cols\n");
      return false;
    }
    *cols_dest = jobj.at("cols");
  }

  if (center_input_dest != nullptr) {
    const njson &ci = jobj.at("center_input");

    const njson::array_t &arr_mass = ci.at("mass");

    for (int i = 0; i < 3; i++) {
      center_input_dest->mass[i] = double(arr_mass[i]) * Ms;
    }

    const njson::array_t &arr_inistate = ci.at("initial_state");

    for (int i = 0; i < 9; i++) {
      center_input_dest->beg_state.position(i) = double(arr_inistate[i]) * rs;
    }
    for (int i = 0; i < 9; i++) {
      center_input_dest->beg_state.velocity(i) =
          double(arr_inistate[i + 9]) * vs;
    }
  }

  if (wind_dest != nullptr) {
    const njson &wind = jobj.at("window");
    const njson &wc = wind.at("center");

    wind_dest->center[0] = wc.at("x");
    wind_dest->center[1] = wc.at("y");

    wind_dest->x_span = wind.at("x_span");
    wind_dest->y_span = wind.at("y_span");
  }

  if (opt_dest != nullptr) {
    const njson &opt = jobj.at("compute_option");

    opt_dest->time_end = double(opt.at("time_end")) * year;
    opt_dest->max_relative_error = opt.at("max_relative_error");
    opt_dest->step_guess = double(opt.at("step_guess")) * year;
  }

  return true;
}