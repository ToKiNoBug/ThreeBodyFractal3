#include "tbf-task.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using std::cout, std::endl;

std::string task_input::tbf_filename(int frameidx) const noexcept {
  std::string name;
  name.reserve(this->tbf_file_prefix.size() + 32);

  name = this->tbf_file_prefix;
  name.append("frame");

  const int total_digits =
      std::ceil(std::log(this->frame_count + 0.1f) / std::log(10.0f));
  char *const dest = name.data() + name.size();
  name.append(total_digits, '\0');

  const int ret = snprintf(dest, name.capacity() - name.size(), "%0*d",
                           total_digits, frameidx);

  if (ret <= 0) {
    cout << "snprintf failed with code " << ret << endl;
    return {};
  }

  name.append(".tbf");

  return name;
}

std::string task_input::png_filename(int frameidx) const noexcept {
  std::string name;
  name.reserve(this->png_file_prefix.size() + 32);

  name = this->png_file_prefix;
  name.append("frame");

  const int total_digits =
      std::ceil(std::log(this->frame_count + 0.1f) / std::log(10.0f));
  char *const dest = name.data() + name.size();
  name.append(total_digits, '\0');

  const int ret = snprintf(dest, name.capacity() - name.size(), "%0*d",
                           total_digits, frameidx);

  if (ret <= 0) {
    cout << "snprintf failed with code " << ret << endl;
    return {};
  }

  name.append(".png");

  return name;
}

bool save_task_to_json(const task_input &ti,
                       std::string_view filename) noexcept {

  using njson = nlohmann::json;

  njson jo;

  jo["zoom_speed"] = ti.zoom_speed;
  jo["frame_count"] = ti.frame_count;

  jo["cpu_threads"] = ti.cpu_threads;
  jo["gpu_threads"] = ti.gpu_threads;
  jo["tbf_file_prefix"] = ti.tbf_file_prefix;
  jo["png_file_prefix"] = ti.png_file_prefix;
  jo["center_source"] = ti.center_source;

  jo["render_json"] = ti.render_json;

  // jo["hide_output"] = ti.reduce_output;
  jo["verbosee"] = ti.verbose;

  std::ofstream ofs(filename.data());

  if (!ofs) {
    cout << "Failed to open file " << filename << endl;
    ofs.close();
    return false;
  }

  try {
    ofs << jo;
  } catch (std::runtime_error re) {
    cout << "Failed to write json. Error detail : " << re.what() << endl;
    ofs.close();
    return false;
  }
  ofs.close();
  return true;
}

bool load_task_from_json(task_input *ti, std::string_view filename) noexcept {
  using njson = nlohmann::json;

  njson jo;

  {
    std::ifstream ifs(filename.data());

    if (!ifs) {
      cout << "Failed to open file " << filename << endl;
      ifs.close();
      return false;
    }

    try {
      ifs >> jo;
    } catch (std::runtime_error re) {
      cout << "Failed to parse json. Error detail : " << re.what() << endl;
      ifs.close();
      return false;
    }

    ifs.close();
  }

  if (!jo.contains("zoom_speed") || !jo.at("zoom_speed").is_number()) {
    cout << "No valid value for \"zoom_speed\"" << endl;
    return false;
  }

  {
    const double zs = jo.at("zoom_speed");
    if (zs <= 0) {
      cout << "Error : zoom_speed = " << zs
           << ", but expected a positive number" << endl;
      return false;
    }

    ti->zoom_speed = zs;
  }

  if (!jo.contains("frame_count") ||
      !jo.at("frame_count").is_number_integer()) {

    cout << "No valid value for \"frame_count\"" << endl;
    return false;
  }

  {
    const int fc = jo.at("frame_count");
    if (fc <= 0) {
      cout << "Error : frame_count = " << fc
           << ", but expected a positive integer" << endl;
      return false;
    }

    ti->frame_count = fc;
  }

  if (!jo.contains("cpu_threads") ||
      !jo.at("cpu_threads").is_number_integer()) {

    cout << "No valid value for \"cpu_threads\"" << endl;
    return false;
  }
  {
    const int ct = jo.at("cpu_threads");
    if (ct <= 0 || ct >= 32768) {
      cout << "Error : cpu_threads = " << ct
           << ", but expected a positive integer no greater than 32767" << endl;
      return false;
    }

    ti->cpu_threads = ct;
  }

  if (!jo.contains("gpu_threads") ||
      !jo.at("gpu_threads").is_number_integer()) {

    cout << "No valid value for \"gpu_threads\"" << endl;
    return false;
  }
  {
    const int gt = jo.at("gpu_threads");
    if (gt < 0 || gt >= 32768) {
      cout << "Error : gpu_threads = " << gt
           << ", but expected a non negative integer no greater than 32767"
           << endl;
      return false;
    }

    ti->gpu_threads = gt;
  }

  if (!jo.contains("center_source") || !jo.at("center_source").is_string()) {
    cout << "No valid value for \"center_source\"" << endl;
    return false;
  }
  {
    std::string cs = jo.at("center_source");
    if (!std::filesystem::is_regular_file(cs)) {
      cout << "Error : center_source = " << cs
           << ", which is not an regular file" << endl;
      return false;
    }

    ti->center_source = std::move(cs);
  }

  if (!jo.contains("tbf_file_prefix") ||
      !jo.at("tbf_file_prefix").is_string()) {
    cout << "No valid value for \"tbf_file_prefix\"" << endl;
    return false;
  }
  ti->tbf_file_prefix = jo.at("tbf_file_prefix");

  if (!jo.contains("png_file_prefix") ||
      !jo.at("png_file_prefix").is_string()) {
    cout << "No valid value for \"png_file_prefix\"" << endl;
    return false;
  }
  ti->png_file_prefix = jo.at("png_file_prefix");

  if (!jo.contains("render_json") || !jo.at("render_json").is_string()) {
    cout << "No valid value for \"render_json\"" << endl;
    return false;
  }

  {
    std::string rj = jo.at("render_json");
    if (rj != "default" && !std::filesystem::is_regular_file(rj)) {
      cout << "Error : render_json = " << rj << ", which is not an regular file"
           << endl;
      return false;
    }

    ti->render_json = std::move(rj);
  }

  if (!jo.contains("verbose") || !jo.at("verbose").is_boolean()) {
    cout << "No valid value for \"verbose\"" << endl;
    return false;
  }

  ti->verbose = jo.at("verbose");

  return true;
}