#include "threebodyfractal.h"

#include <stdio.h>

#include <atomic>
#include <mutex>
#include <thread>

void libthreebody::compute_many(const input_t *const src, result_t *const dest,
                                const uint64_t count,
                                const compute_options &opt) noexcept {
  using namespace libthreebody;

  constexpr int batch_size = 1024;

#pragma omp parallel for schedule(dynamic)
  for (int batch_idx = 0; batch_idx < (int)std::ceil(float(count) / batch_size);
       batch_idx++) {
    for (int idx = batch_idx * batch_size;
         idx < std::max((batch_idx + 1) * batch_size, int(count)); idx++) {
      simulate_2(src[idx], opt, dest + idx);
    }
  }
}

void libthreebody::compute_frame(
    const input_t &center_input, const fractal_utils::center_wind<double> &wind,
    const compute_options &opt,
    fractal_utils::fractal_map *const dest) noexcept {
  if (dest->element_bytes != sizeof(libthreebody::result_t)) {
    printf(
        "\nError in function libthreebody::compute_frame : element size "
        "mismatch. dest->element_byte = %u but "
        "sizeof(libthreebody::result_t) is %llu\n",
        dest->element_bytes, sizeof(libthreebody::result_t));
    return;
  }

  using namespace libthreebody;

  const auto left_top_pos = wind.left_top_corner();

  const double y_per_row = -wind.y_span / dest->rows;
  const double x_per_col = wind.x_span / dest->cols;

  std::atomic_int finished_rows = 0;
  std::mutex lock;

#pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < dest->rows; r++) {
    std::array<double, 2> pos;
    pos[1] = left_top_pos[1] + y_per_row * r;

    input_t input = center_input;

    for (int c = 0; c < dest->cols; c++) {
      pos[0] = left_top_pos[0] + x_per_col * c;
      input.mass[1] = center_input.mass[1] * std::pow(10.0, pos[0]);
      input.mass[2] = center_input.mass[2] * std::pow(10.0, pos[1]);

      simulate(input, opt, &dest->at<libthreebody::result_t>(r, c));
    }
    finished_rows++;

    lock.lock();
    printf("%i tasks finished.\n", int(finished_rows * dest->cols));
    lock.unlock();
  }
}
