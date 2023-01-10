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

const libthreebody::color_map_all libthreebody::default_color_map_0{
    std::array<std::array<float, 2>, 3>{std::array<float, 2>{0.0f, 0.333f},
                                        std::array<float, 2>{0.333f, 0.6667f},
                                        std::array<float, 2>{0.667f, 1.0f}},

    std::array<std::array<float, 2>, 3>{std::array<float, 2>{0.0f, 1.0f},
                                        std::array<float, 2>{0.0f, 1.0f},
                                        std::array<float, 2>{0.0f, 1.0f}},

    std::array<fractal_utils::color_series, 3>{
        fractal_utils::color_series::jet, fractal_utils::color_series::jet,
        fractal_utils::color_series::jet},

    std::array<fractal_utils::color_series, 3>{
        fractal_utils::color_series::pink, fractal_utils::color_series::pink,
        fractal_utils::color_series::pink},
    std::array<uint8_t, 3>{0xFF, 0xFF, 0xFF},
    std::array<uint8_t, 3>{0xFF, 0xFF, 0xFF},
    std::array<render_method, 3>{render_method::collide_time,
                                 render_method::collide_time,
                                 render_method::collide_time},
    std::array<render_method, 3>{render_method::triangle,
                                 render_method::triangle,
                                 render_method::triangle}};

void libthreebody::color_by_all(const result_t *const src, float *const buffer,
                                fractal_utils::pixel_RGB *const dest_u8c3,
                                int num, double max_time,
                                const color_map_all &color_map) noexcept {
  std::array<double, 3> distance_2;
  std::array<float, 3> distance;
  for (int i = 0; i < num; i++) {
    int idx = compute_distance_idx(src[i].end_state.position, &distance_2);

    const bool collide = src[i].end_time < max_time;

    const auto &range = color_map.range(collide, idx);
    const auto cs = color_map.color_serie(collide, idx);

    float val;
    if (collide) {
      val = src[i].end_time / max_time;
    } else {
      for (int j = 0; j < 3; j++) {
        distance[j] = std::sqrt(float(distance_2[j]));
      }
      val = distance[2] / (distance[0] + distance[1]);
    }
    dest_u8c3[i] =
        fractal_utils::color_u8c3(val * (range[1] - range[0]) + range[0], cs);
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

    for (int j = 0; j < 3; j++) {
      distance[j] = std::sqrt(float(distance_2[j]));
    }

    buffer[i] = distance[2] / (distance[0] + distance[1]);
  }

  fractal_utils::color_u8c3_many(buffer, cs, num, dest_u8c3);
}

#include <limits>

struct range_t {
  float min{std::numeric_limits<float>::infinity()};
  float max{-std::numeric_limits<float>::infinity()};
};

bool libthreebody::render_universial(
    const fractal_utils::fractal_map &map_result,
    const std::array<int, 2> &skip_rows_cols, void *const _voidp_buffer,
    size_t buffer_capacity, fractal_utils::fractal_map *const img_u8c3,
    double max_time, const color_map_all &color_map) noexcept {
  const size_t row_beg = skip_rows_cols[0];
  const size_t row_end = map_result.rows - skip_rows_cols[0];

  const size_t col_beg = skip_rows_cols[1];
  const size_t col_end = map_result.cols - skip_rows_cols[1];

  if (row_end < row_beg + 2 || col_end < col_beg + 2) {
    printf("\nError : image too small.\n");
    return false;
  }

  if (buffer_capacity <
      map_result.element_count() * (sizeof(float) + sizeof(uint8_t))) {
    printf(
        "\nError : libthreebody::render_universial failed : insuffcient "
        "memory.\n");
    return false;
  }

  uint8_t *const _u8p_buffer = reinterpret_cast<uint8_t *>(_voidp_buffer);

  fractal_utils::fractal_map map_float(map_result.rows, map_result.cols,
                                       sizeof(float), _u8p_buffer);

  fractal_utils::fractal_map map_u8(map_result.rows, map_result.cols,
                                    sizeof(uint8_t),
                                    _u8p_buffer + map_float.byte_count());

  // offset=idx_3*2+collide;
  // collide=offset&0b1;
  // idx_3=offset>>1;

  std::array<range_t, 6> range_arr;
  std::array<fractal_utils::color_source_t, 6> color_source_t_arr;

  for (uint8_t offset = 0; offset < 6; offset++) {
    const bool collide = offset & 0b1;
    const int idx_3 = offset >> 1;

    color_source_t_arr[offset] =
        fractal_utils::color_source(color_map.color_serie(collide, idx_3));
  }

  std::array<double, 3> distance_2;
  std::array<float, 3> distance;

  for (size_t r = row_beg; r < row_end; r++) {
    for (size_t c = col_beg; c < col_end; c++) {
      const bool collide = map_result.at<result_t>(r, c).end_time < max_time;
      const int idx_3 = compute_distance_idx(
          map_result.at<result_t>(r, c).end_state.position, &distance_2);

      float val;

      switch (color_map.render_method_(collide, idx_3)) {
        case render_method::collide_binary:
          val = collide;
          break;
        case render_method::collide_time:
          val = map_result.at<result_t>(r, c).end_time / max_time;
          break;
        case render_method::end_distance:
          val = idx_3 / 2.0f;
          break;
        case render_method::triangle: {
          for (int j = 0; j < 3; j++) {
            distance[j] = std::sqrt(distance_2[j]);
          }
          val = distance[2] / (distance[0] + distance[1]);
          break;
        }
      }
      // end switch

      const uint8_t nid = color_map.normalize_id(collide, idx_3);
      if (nid < 6) {
        range_arr[nid].max = std::max(range_arr[nid].max, val);
        range_arr[nid].min = std::min(range_arr[nid].min, val);
      }

      uint8_t offset = idx_3 * 2 + collide;
      map_float.at<float>(r, c) = val;
      map_u8.at<uint8_t>(r, c) = offset;
    }
  }

  for (size_t r = row_beg; r < row_end; r++) {
    for (size_t c = col_beg; c < col_end; c++) {
      const uint8_t offset = map_u8.at<uint8_t>(r, c);
      const bool collide = offset & 0b1;
      const int idx_3 = offset >> 1;

      const uint8_t nid = color_map.normalize_id(collide, idx_3);

      float &val = map_float.at<float>(r, c);
      if (nid < 6) {
        const float min = range_arr[nid].min;
        const float max = range_arr[nid].max;

        val = (val - min) / (max - min) + min;
      }
      const auto &range = color_map.range(collide, idx_3);
      val = (range[1] - range[0]) * val + range[0];

      img_u8c3->at<fractal_utils::pixel_RGB>(r, c) =
          fractal_utils::color_u8c3(val, color_source_t_arr[offset]);
    }
  }

  return true;
}

#include <fstream>
#include <nlohmann/json.hpp>

bool parse_range_cs_pair(const nlohmann::json &object,
                         std::array<float, 2> *range,
                         fractal_utils::color_series *cs) noexcept {
  using njson = nlohmann::json;

  if (!object.contains("range") || !object.at("range").is_array()) {
    printf("\nError : no valid value for float array \"range\".\n");
    return false;
  }

  const njson::array_t &range_arr = object.at("range");

  if (range_arr.size() != 2 || !range_arr.front().is_number()) {
    printf(
        "\nError : invalid value for float array \"range\" : element is not "
        "number.\n");
    return false;
  }

  for (int i = 0; i < 2; i++) {
    range->at(i) = range_arr[i];
    if (range->at(i) < 0.0f || range->at(i) > 1.0f) {
      printf(
          "\nError : invalid value for float array \"range\" : element at "
          "index %i is %F, which goes out of range [0,1].\n",
          i, range->at(i));
      return false;
    }
  }

  if (!object.contains("color_serie") ||
      !object.at("color_serie").is_string()) {
    printf("\nError : no valid value for string \"color_serie\".\n");
    return false;
  }

  const std::string cs_str = object.at("color_serie");

  bool ok = true;

  *cs = fractal_utils::color_series_str_to_enum(cs_str.data(), &ok);
  if (!ok) {
    printf("\nError : unknown color serie named \"%s\"\n", cs_str.data());
    return false;
  }

  // printf("color serie = %s\n", fractal_utils::color_series_enum_to_str(*cs));
#warning here
  return true;
}

// #include <iostream>

bool libthreebody::load_color_map_all_from_file(
    const char *const filename, color_map_all *const dest) noexcept {
  using njson = nlohmann::json;
  njson json;
  try {
    std::ifstream(filename) >> json;
  } catch (...) {
    printf("\nError : nlohmann json failed to parse json file : %s\n",
           filename);
    return false;
  }

  if (!json.contains("color_map_all") ||
      !json.at("color_map_all").is_object()) {
    printf("\nError : no valid value for object color_map_all.\n");
    return false;
  }

  const njson &obj = json.at("color_map_all");

  // std::cout << obj << std::endl;

  const std::array<const char *, 2> keys{"collide", "nocollide"};

  for (int kidx = 0; kidx < 2; kidx++) {
    if (!obj.contains(keys[kidx]) || !obj.at(keys[kidx]).is_array()) {
      printf(
          "\nError : no valid value for key \"%s\" : expected an array of "
          "objects\n",
          keys[kidx]);
      return false;
    }

    const njson::array_t &arr = obj.at(keys[kidx]);

    if (arr.size() != 3 || !arr.front().is_object()) {
      printf(
          "\nError : invalid value for array named \"%s\" : size should be 3 "
          "and elements should be objects\n",
          keys[kidx]);
      return false;
    }
  }

  for (int idx = 0; idx < 3; idx++) {
    if (!parse_range_cs_pair(obj.at("collide")[idx],
                             &dest->float_range_lut_collide[idx],
                             &dest->cs_lut_collide[idx])) {
      printf("\nError : failed to parse color_map_all.\n");
      return false;
    }
    if (!parse_range_cs_pair(obj.at("nocollide")[idx],
                             &dest->float_range_lut_nocollide[idx],
                             &dest->cs_lut_nocollide[idx])) {
      printf("\nError : failed to parse color_map_all.\n");
      return false;
    }
  }

  return true;
}