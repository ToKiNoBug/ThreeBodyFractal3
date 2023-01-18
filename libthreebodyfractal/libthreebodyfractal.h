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

bool save_fractal_basical_information_binary(
    std::string_view filename, const input_t &center_input,
    const fractal_utils::center_wind<double> &wind, const compute_options &opt,
    const std::array<size_t, 2> &size_rc) noexcept;

bool save_fractal_basical_information_json(
    std::string_view filename, const input_t &center_input,
    const fractal_utils::center_wind<double> &wind, const compute_options &opt,
    const std::array<size_t, 2> &size_rc) noexcept;

bool load_fractal_basical_information_json(
    std::string_view filename, size_t *const rows_dest = nullptr,
    size_t *const cols_dest = nullptr,
    input_t *const center_input_dest = nullptr,
    fractal_utils::center_wind<double> *const wind_dest = nullptr,
    compute_options *const opt_dest = nullptr) noexcept;

bool load_fractal_basical_information_nbt(
    std::string_view filename, size_t *const rows_dest = nullptr,
    size_t *const cols_dest = nullptr,
    input_t *const center_input_dest = nullptr,
    fractal_utils::center_wind<double> *const wind_dest = nullptr,
    compute_options *const opt_dest = nullptr) noexcept;

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

enum class render_method : uint8_t {
  collide_time,
  end_distance,
  collide_binary,
  triangle
};

render_method render_method_str_to_enum(const char *const str,
                                        bool *ok = nullptr) noexcept;

struct color_map_all {
  std::array<std::array<float, 2>, 3> float_range_lut_collide;
  std::array<std::array<float, 2>, 3> float_range_lut_nocollide;
  std::array<fractal_utils::color_series, 3> cs_lut_collide;
  std::array<fractal_utils::color_series, 3> cs_lut_nocollide;
  // id with 0xFF means do not normalize
  std::array<uint8_t, 3> normalize_id_collide;
  // id with 0xFF means do not normalize
  std::array<uint8_t, 3> normalize_id_nocollide;
  std::array<render_method, 3> method_collide;
  std::array<render_method, 3> method_nocollide;

  inline const std::array<float, 2> &range(bool collide,
                                           int idx) const noexcept {
    if (collide) {
      return float_range_lut_collide[idx];
    }
    return float_range_lut_nocollide[idx];
  }

  inline fractal_utils::color_series color_serie(bool collide,
                                                 int idx) const noexcept {
    if (collide) {
      return cs_lut_collide[idx];
    }
    return cs_lut_nocollide[idx];
  }

  inline uint8_t normalize_id(bool collide, int idx) const noexcept {
    if (collide) {
      return normalize_id_collide[idx];
    }
    return normalize_id_nocollide[idx];
  }

  inline render_method render_method_(bool collide, int idx) const noexcept {
    if (collide) {
      return method_collide[idx];
    }
    return method_nocollide[idx];
  }
};

extern const color_map_all default_color_map_0;
extern const color_map_all default_color_map_1;

[[deprecated]] void
color_by_all(const result_t *const src, float *const buffer,
             fractal_utils::pixel_RGB *const dest_u8c3, int num,
             double max_time,
             const color_map_all &color_map = default_color_map_0) noexcept;

bool load_color_map_all_from_file(const char *const filename,
                                  color_map_all *const dest) noexcept;

bool render_universial(
    const fractal_utils::fractal_map &map_result,
    const std::array<int, 2> &skip_rows_cols, void *const buffer,
    size_t buffer_capacity, fractal_utils::fractal_map *const img_u8c3,
    double max_time,
    const color_map_all &color_map = default_color_map_0) noexcept;

} // namespace libthreebody

#endif // THREEBODYFRACTAL3_THREEBODYFRACTAL_H