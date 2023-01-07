#ifndef THREEBODYFRACTAL3_THREEBODYFRACTAL_H
#define THREEBODYFRACTAL3_THREEBODYFRACTAL_H

#include <fractal_utils/core_utils.h>

#include <string>
#include <string_view>

#include "libthreebody.h"
#include "memory_resource.h"

namespace libthreebody {

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

}  // namespace libthreebody

#endif  // THREEBODYFRACTAL3_THREEBODYFRACTAL_H