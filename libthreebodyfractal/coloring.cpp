#include <immintrin.h>

#include "libthreebodyfractal.h"

void libthreebody::color_by_end_age_u8c3(
    const result_t *const src, float *const buffer,
    fractal_utils::pixel_RGB *const dest_u8c3, int num, double max_time,
    bool invert_float, fractal_utils::color_series cs) noexcept {
  for (int i = 0; i < num; i++) {
    if (invert_float) {
      buffer[i] = 1.0f - float(src[i].end_time) / float(max_time);
    } else {
      buffer[i] = float(src[i].end_time) / float(max_time);
    }
  }

  fractal_utils::color_u8c3_many(buffer, cs, num, dest_u8c3);
}

inline int compute_distance_idx(const Eigen::Array33d &pos) noexcept {
  std::array<double, 3> distance_2{0, 0, 0};

  for (int r = 0; r < 3; r++) {
    distance_2[0] += (pos(r, 0) - pos(r, 1)) * (pos(r, 0) - pos(r, 1));
    distance_2[1] += (pos(r, 1) - pos(r, 2)) * (pos(r, 1) - pos(r, 2));
    distance_2[2] += (pos(r, 0) - pos(r, 2)) * (pos(r, 0) - pos(r, 2));
  }

  int max_idx = 0;
  for (int i = 1; i < 3; i++) {
    if (distance_2[i] > distance_2[max_idx]) {
      max_idx = i;
    }
  }

  return max_idx;
}

void libthreebody::color_by_end_distance_u8c3(
    const result_t *const src, fractal_utils::pixel_RGB *const dest_u8c3,
    int num,
    const std::array<fractal_utils::pixel_RGB, 3> &color_arr) noexcept {
  for (int i = 0; i < num; i++) {
    int idx = compute_distance_idx(src[i].end_state.position);
    dest_u8c3[i] = color_arr[idx];
  }
}

void libthreebody::color_by_collide_u8c3(
    const result_t *const src, fractal_utils::pixel_RGB *const dest_u8c3,
    int num, double max_time,
    const std::array<fractal_utils::pixel_RGB, 2> &color_arr) noexcept {
  for (int i = 0; i < num; i++) {
    bool collide = src[i].end_time < max_time;

    int idx = collide;

    dest_u8c3[i] = color_arr[idx];
  }
}