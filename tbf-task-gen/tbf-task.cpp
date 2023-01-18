#include "tbf-task.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>


bool save_task_to_json(const task_input &ti,
                       std::string_view filename) noexcept {

  using njson = nlohmann::json;

  njson jo;

  jo["zoom_speed"] = ti.zoom_speed;
  jo["frame_count"] = ti.frame_count;

  jo["cpu_threads"] = ti.cpu_threads;
  jo["gpu_threads"] = ti.gpu_threads;
  jo["tbf_file_prefix"] = ti.tbf_file_prefix;
  jo["center_source"] = ti.center_source;

  std::ofstream ofs(filename.data());

  if (!ofs) {
    std::cout << "Failed to open file " << filename << std::endl;
    ofs.close();
    return false;
  }
  try {

    ofs << jo;
  } catch (std::runtime_error re) {
    std::cout << "Failed to write json. Error detail : " << re.what()
              << std::endl;
    ofs.close();
    return false;
  }
  ofs.close();
  return true;
}