#ifndef THREEBODYFRACTAL3_TBF_TASK_H
#define THREEBODYFRACTAL3_TBF_TASK_H

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <string_view>

struct task_input {
  double zoom_speed;
  int frame_count;
  int16_t cpu_threads;
  int16_t gpu_threads;

  std::string center_source;
  std::string tbf_file_prefix;
};

bool save_task_to_json(const task_input &ti,
                       std::string_view filename) noexcept;

bool load_task_from_json(task_input *ti, std::string_view filename) noexcept;
#endif // THREEBODYFRACTAL3_TBF_TASK_H