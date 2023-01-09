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

inline int compute_distance_idx(
    const Eigen::Array33d &pos,
    std::array<double, 3> *const dest = nullptr) noexcept {
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

  if (dest != nullptr) {
    *dest = distance_2;
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

const libthreebody::render_color_map libthreebody::default_color_map_0{
    std::array<std::array<float, 2>, 3>{std::array<std::array<float, 2>, 3>{
        std::array<float, 2>{0.0f, 0.333f},
        std::array<float, 2>{0.333f, 0.6667f},
        std::array<float, 2>{0.667f, 1.0f}}},
    std::array<fractal_utils::color_series, 3>{
        fractal_utils::color_series::jet, fractal_utils::color_series::jet,
        fractal_utils::color_series::jet},
    std::array<fractal_utils::pixel_RGB, 3>{
        fractal_utils::pixel_RGB{0xfc, 0xb1, 0xb1},
        fractal_utils::pixel_RGB{0xf0, 0xf6, 0x96},
        fractal_utils::pixel_RGB{0x96, 0xf7, 0xd2}}};

void libthreebody::color_by_all(const result_t *const src, float *const buffer,
                                fractal_utils::pixel_RGB *const dest_u8c3,
                                int num, double max_time,
                                const render_color_map &color_map) noexcept {
  for (int i = 0; i < num; i++) {
    int idx = compute_distance_idx(src[i].end_state.position);
    if (src->end_time < max_time) {
      const auto &range = color_map.float_range_lut_collide[idx];
      const auto cs = color_map.cs_lut_collide[idx];
      float val = src->end_time / max_time;

      dest_u8c3[i] =
          fractal_utils::color_u8c3(val * (range[1] - range[0]) + range[0], cs);
    } else {
      dest_u8c3[i] = color_map.color_no_collide[idx];
    }
  }
}

void libthreebody::color_by_triangle(const result_t *const src,
                                     float *const buffer,
                                     fractal_utils::pixel_RGB *const dest_u8c3,
                                     int num,
                                     fractal_utils::color_series cs) noexcept {
  std::array<double, 3> distance_2;
  std::array<float, 3> distance;
  for (int i = 0; i < num; i++) {
    compute_distance_idx(src[i].end_state.position, &distance_2);

    for (int j = 0; i < 3; j++) {
      distance[j] = std::sqrt(float(distance_2[j]));
    }

    buffer[i] = distance[2] / (distance[0] + distance[1]);
  }

  fractal_utils::color_u8c3_many(buffer, cs, num, dest_u8c3);
}