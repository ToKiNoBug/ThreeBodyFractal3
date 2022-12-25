#ifndef THREEBODYFRACTAL3_THREEBODYFRACTAL_H
#define THREEBODYFRACTAL3_THREEBODYFRACTAL_H

#include <fractal_utils/core_utils.h>

#include <string_view>

#include "libthreebody.h"

namespace libthreebody {

void compute_many(const input_t *const src, result_t *const dest,
                  const uint64_t count, const compute_options &opt) noexcept;

void compute_frame(const input_t &center_input,
                   const fractal_utils::center_wind<double> &wind,
                   const compute_options &opt,
                   fractal_utils::fractal_map *const dest_result) noexcept;

enum fractal_binfile_tag : int64_t {
  basical_information,
  matrix_end_state,
  matrix_end_energy,
  matrix_collide_time,
  matrix_iterate_time,
  matrix_iterate_fail_time
};

bool save_fractal_bin_file(std::string_view filename,
                           const input_t &center_input,
                           const fractal_utils::center_wind<double> &wind,
                           const compute_options &opt,
                           const fractal_utils::fractal_map &mat_result,
                           void *const buffer,
                           const size_t buffer_bytes) noexcept;

}  // namespace libthreebody

#endif  // THREEBODYFRACTAL3_THREEBODYFRACTAL_H