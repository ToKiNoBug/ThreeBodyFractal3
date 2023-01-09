#include "libthreebodyfractal.h"

#include <stdio.h>

#include <atomic>
#include <mutex>
#include <thread>

#include "../libcudathreebody/libcudathreebody.h"

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

void libthreebody::compute_frame(const input_t &center_input,
                                 const fractal_utils::center_wind<double> &wind,
                                 const compute_options &opt,
                                 fractal_utils::fractal_map *const dest,
                                 bool display_progress) noexcept {
  if (dest->element_bytes != sizeof(libthreebody::result_t)) {
    printf(
        "\nError in function libthreebody::compute_frame : element size "
        "mismatch. dest->element_byte = %u but "
        "sizeof(libthreebody::result_t) is %zu\n",
        dest->element_bytes, sizeof(libthreebody::result_t));
    return;
  }

  using namespace libthreebody;

  const auto left_top_pos = wind.left_top_corner();

  const double y_per_row = -wind.y_span / dest->rows;
  const double x_per_col = wind.x_span / dest->cols;

  std::atomic_int finished_rows(0);
  std::mutex lock;
  if (display_progress) {
    printf("\r[ %i / %i ] %f%% tasks finished.",
           int(finished_rows * dest->cols), int(dest->element_count()),
           float(100 * finished_rows * dest->cols) / dest->element_count());
  }

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
    if (display_progress && lock.try_lock()) {
      printf("\r[ %i / %i ] %f%% tasks finished.",
             int(finished_rows * dest->cols), int(dest->element_count()),
             float(100 * finished_rows * dest->cols) / dest->element_count());
      lock.unlock();
    }
  }

  printf("\n");
}

void libthreebody::compute_frame_cpu_and_gpu(
    const input_t &center_input, const fractal_utils::center_wind<double> &wind,
    const compute_options &opt, fractal_utils::fractal_map *const dest,
    gpu_mem_allocator *const allocator, bool display_progress) noexcept {
  if (dest->element_bytes != sizeof(libthreebody::result_t)) {
    printf(
        "\nError in function libthreebody::compute_frame : element size "
        "mismatch. dest->element_byte = %u but "
        "sizeof(libthreebody::result_t) is %zu\n",
        dest->element_bytes, sizeof(libthreebody::result_t));
    return;
  }

  using namespace libthreebody;

  const auto left_top_pos = wind.left_top_corner();

  const double y_per_row = -wind.y_span / dest->rows;
  const double x_per_col = wind.x_span / dest->cols;

  std::atomic_int finished_rows(0);
  std::mutex lock;

#pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < dest->rows; r++) {
    std::array<double, 2> pos;
    pos[1] = left_top_pos[1] + y_per_row * r;

    input_t input = center_input;

    const gpu_memory_resource_t *const res = allocator->allocate();

    if (res == nullptr) {
      for (int c = 0; c < dest->cols; c++) {
        pos[0] = left_top_pos[0] + x_per_col * c;
        input.mass[1] = center_input.mass[1] * std::pow(10.0, pos[0]);
        input.mass[2] = center_input.mass[2] * std::pow(10.0, pos[1]);

        simulate(input, opt, &dest->at<libthreebody::result_t>(r, c));
      }
    } else {
      for (int c = 0; c < dest->cols; c++) {
        pos[0] = left_top_pos[0] + x_per_col * c;
        input.mass[1] = center_input.mass[1] * std::pow(10.0, pos[0]);
        input.mass[2] = center_input.mass[2] * std::pow(10.0, pos[1]);

        reinterpret_cast<input_t *>(res->host_input_buffer)[c] = input;
      }

      libcudathreebody::run_cuda_simulations(
          reinterpret_cast<input_t *>(res->host_input_buffer),
          &dest->at<result_t>(r, 0), res->device_mem_input,
          res->device_mem_result, dest->cols, opt);

      allocator->deallocate(res);
    }

    finished_rows++;
    if (display_progress && lock.try_lock()) {
      printf("\r[ %i / %i ] %f%% tasks finished.",
             int(finished_rows * dest->cols), int(dest->element_count()),
             float(100 * finished_rows * dest->cols) / dest->element_count());
      lock.unlock();
    }
  }

  printf("\n");
}

inline uint8_t char_to_u4(char c) noexcept {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }

  if (c >= 'A' && c <= 'F') {
    return c - 'A';
  }

  if (c >= 'a' && c <= 'f') {
    return c - 'a';
  }

  return 0xFF;
}

bool libthreebody::hex_to_binary(const char *hex, void *__binary) noexcept {
  if (hex[0] == '0' && (hex[1] == 'X' || hex[1] == 'x')) {
    hex += 2;
  }

  const size_t hex_strlen = std::strlen(hex);
  if (hex_strlen % 2 != 0) {
    printf(
        "\nError : failed to convert hex to binary : hex string length is %zu, "
        "but expected even number.\n",
        hex_strlen);
    return false;
  }

  uint8_t *binary = reinterpret_cast<uint8_t *>(__binary);

  while (true) {
    if (hex[0] == '\0') {
      break;
    }

    const uint8_t high_4 = char_to_u4(*(hex++));
    if (high_4 >= 16) {
      printf("\nError : invalid char in hex string %c\n", *(hex - 1));
      return false;
    }
    const uint8_t low_4 = char_to_u4(*(hex++));
    if (low_4 >= 16) {
      printf("\nError : invalid char in hex string %c\n", *(hex - 1));
      return false;
    }

    *(binary++) = (high_4 << 4) | (low_4 & 0b1111);
  }

  return true;
}