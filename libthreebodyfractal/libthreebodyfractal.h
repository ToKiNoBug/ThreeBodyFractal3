#ifndef THREEBODYFRACTAL3_THREEBODYFRACTAL_H
#define THREEBODYFRACTAL3_THREEBODYFRACTAL_H

#include <fractal_utils/core_utils.h>

#include <string>
#include <string_view>

#include "libthreebody.h"
#include "memory_resource.h"

namespace libthreebody {

bool hex_to_binary(const char *hex, void *binary) noexcept;

void compute_many(const input_t *const src, result_t *const dest,
                  const uint64_t count, const compute_options &opt) noexcept;

void compute_frame(const input_t &center_input,
                   const fractal_utils::center_wind<double> &wind,
                   const compute_options &opt,
                   fractal_utils::fractal_map *const dest_result,
                   bool display_progress = true) noexcept;

void compute_frame_cpu_and_gpu(const input_t &center_input,
                               const fractal_utils::center_wind<double> &wind,
                               const compute_options &opt,
                               fractal_utils::fractal_map *const dest_result,
                               gpu_mem_allocator *const allocator,
                               bool display_progress = true) noexcept;

enum fractal_binfile_tag : int64_t {
  basical_information = 1,
  matrix_end_state = 2,
  matrix_end_energy = 3,
  matrix_collide_time = 4,
  matrix_iterate_time = 5,
  matrix_iterate_fail_time = 6
};

bool save_fractal_bin_file(std::string_view filename,
                           const input_t &center_input,
                           const fractal_utils::center_wind<double> &wind,
                           const compute_options &opt,
                           const fractal_utils::fractal_map &mat_result,
                           void *const buffer,
                           const size_t buffer_bytes) noexcept;

bool fractal_bin_file_get_information(
    const fractal_utils::binfile &binfile, size_t *const rows_dest = nullptr,
    size_t *const cols_dest = nullptr,
    input_t *const center_input_dest = nullptr,
    fractal_utils::center_wind<double> *const wind_dest = nullptr,
    compute_options *const opt_dest = nullptr) noexcept;

bool fractal_bin_file_get_end_state(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_state_dest,
    const bool examine_map_size = false) noexcept;

bool fractal_bin_file_get_end_energy(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_energy_dest,
    const bool examine_map_size = false) noexcept;

bool fractal_bin_file_get_collide_time(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_time_dest,
    const bool examine_map_size = false) noexcept;

bool fractal_bin_file_get_iterate_time(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_iterate_time_dest,
    const bool examine_map_size = false) noexcept;

bool fractal_bin_file_get_iterate_fail_time(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_iterate_fail_time_dest,
    const bool examine_map_size = false) noexcept;

bool fractal_bin_file_get_result(const fractal_utils::binfile &binfile,
                                 fractal_utils::fractal_map *const result_dest,
                                 void *buffer, size_t buffer_capacity,
                                 const bool examine_map_size = false) noexcept;

void color_by_end_age_u8c3(const result_t *const src, float *const buffer,
                           fractal_utils::pixel_RGB *const dest_u8c3, int num,
                           double max_time, bool invert_float = true,
                           fractal_utils::color_series cs =
                               fractal_utils::color_series::parula) noexcept;

void color_by_end_distance_u8c3(
    const result_t *const src, fractal_utils::pixel_RGB *const dest_u8c3,
    int num,
    const std::array<fractal_utils::pixel_RGB, 3> &color_arr = {
        fractal_utils::pixel_RGB{62, 38, 168},
        fractal_utils::pixel_RGB{249, 251, 21},
        fractal_utils::pixel_RGB{69, 203, 137}}) noexcept;

void color_by_collide_u8c3(
    const result_t *const src, fractal_utils::pixel_RGB *const dest_u8c3,
    int num, double max_time,
    const std::array<fractal_utils::pixel_RGB, 2> &color_arr = {
        fractal_utils::pixel_RGB{62, 38, 168},
        fractal_utils::pixel_RGB{69, 203, 137}}) noexcept;

void color_by_end_distance_and_age_u8c3(
    const result_t *const src, float *const buffer,
    fractal_utils::pixel_RGB *const dest_u8c3, int num, double max_time,
    bool invert_float = true,
    fractal_utils::color_series cs =
        fractal_utils::color_series::parula) noexcept;

struct render_color_map {
  std::array<std::array<std::array<float, 2>, 3>, 2> float_range_lut;
  std::array<std::array<fractal_utils::color_series, 3>, 2> cs_lut;

  inline const std::array<float, 2> &float_range(
      bool collide, int beg_state_idx) const noexcept {
    return float_range_lut[int(collide)][beg_state_idx];
  }

  inline fractal_utils::color_series color_serie_at(
      bool collide, int beg_state_idx) const noexcept {
    return cs_lut[int(collide)][beg_state_idx];
  }

  bool is_color_series_single() const noexcept {
    fractal_utils::color_series cs = cs_lut[0][0];

    for (int i = 0; i < cs_lut.size(); i++) {
      for (int j = 0; j < cs_lut[i].size(); j++) {
        if (cs != cs_lut[i][j]) {
          return false;
        }
      }
    }

    return true;
  }
};

extern const render_color_map default_color_map_0;

void color_by_all(
    const result_t *const src, float *const buffer,
    fractal_utils::pixel_RGB *const dest_u8c3, int num, double max_time,
    const render_color_map &color_map = default_color_map_0) noexcept;

}  // namespace libthreebody

#endif  // THREEBODYFRACTAL3_THREEBODYFRACTAL_H