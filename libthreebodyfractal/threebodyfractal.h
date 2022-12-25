#ifndef THREEBODYFRACTAL3_THREEBODYFRACTAL_H
#define THREEBODYFRACTAL3_THREEBODYFRACTAL_H

#include "libthreebody.h"

#include <fractal_utils/core_utils.h>

namespace libthreebody {

void compute_many(const input_t *const src, result_t *const dest,
                  const uint64_t count, const compute_options &opt) noexcept;

void compute_frame(const input_t &center_input,
                   const fractal_utils::center_wind<double> &wind,
                   const compute_options &opt,
                   fractal_utils::fractal_map *const dest_result) noexcept;

} // namespace libthreebody

#endif // THREEBODYFRACTAL3_THREEBODYFRACTAL_H