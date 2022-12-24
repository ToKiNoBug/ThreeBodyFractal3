#include "threebodyfractal.h"

void libthreebody::compute_many(const input_t *const src, result_t *const dest,
                                const uint64_t count,
                                const compute_options &opt) {
  using namespace libthreebody;

  constexpr int batch_size = 1024;

#pragma omp parallel for schedule(dynamic)
  for (int batch_idx = 0; batch_idx < (int)std::ceil(float(count) / batch_size);
       batch_idx++) {
    for (int idx = batch_idx * batch_size;
         idx < std::max((batch_idx + 1) * batch_size, int(count)); idx++) {
      simulate(src[idx], opt, dest + idx);
    }
  }
}