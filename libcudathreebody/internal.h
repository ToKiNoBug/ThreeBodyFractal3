#ifndef THREEBODYFRACTAL3_LIBCUDATHREEBODY_INTERNAL_H
#define THREEBODYFRACTAL3_LIBCUDATHREEBODY_INTERNAL_H

#include "../libthreebody/libthreebody.h"

namespace libcudathreebody {
constexpr double __constant__ G = ::libthreebody::G;
constexpr double __constant__ Ms = ::libthreebody::year;
constexpr double __constant__ omega_s = ::libthreebody::omega_s;
constexpr double __constant__ rho = ::libthreebody::rho;
constexpr double __constant__ rs = ::libthreebody::rs;
constexpr double __constant__ vs = ::libthreebody::vs;
constexpr double __constant__ as = ::libthreebody::as;
constexpr double __constant__ year = ::libthreebody::year;

using mass_t = ::libthreebody::mass_t;
using state_t = ::libthreebody::state_t;
using input_t = ::libthreebody::input_t;
using compute_options = ::libthreebody::compute_options;
using result_t = ::libthreebody::result_t;

double __device__ compute_kinetic(const Eigen::Array33d &velocity,
                                  const mass_t &mass) noexcept;
double __device__ compute_potential(const Eigen::Array33d &position,
                                    const mass_t &mass) noexcept;
void __device__ compute_acclerate(const Eigen::Array33d &position,
                                  const mass_t &mass,
                                  Eigen::Array33d *const acclerate) noexcept;
void __device__ compute_potential_acclerate(
    const Eigen::Array33d &position, const mass_t &mass,
    double *const potential, Eigen::Array33d *const acclerate) noexcept;

inline double __device__ compute_energy(const state_t &s,
                                        const mass_t &mass) noexcept {
  return compute_kinetic(s.velocity, mass) +
         compute_potential(s.position, mass);
}

void __device__ rk4_update_state(double step, const double *y_nd,
                                 const double *k1d, const double *k2d,
                                 const double *k3d, const double *k4d,
                                 double *y_n1d) noexcept;

void __device__ rk4(const state_t &y_n, const mass_t &mass, const double step,
                    state_t *const y_n1) noexcept;

void __device__ rk4_2(const state_t &y_n, const mass_t mass, const double step,
                      state_t *const y_n1,
                      const Eigen::Array33d &acclerate_of_y_n) noexcept;

void __global__ simulate_10(const input_t *const inputs,
                            const compute_options opt, result_t *const results);

template <int task_num>
void __global__ simulate_N(const input_t *const inputs,
                           const compute_options opt, result_t *const results) {
  constexpr int blockDim_x_val = task_num * 3;
  if (blockDim.x != blockDim_x_val) {
    return;
  }

  const int task_offset = threadIdx.x / 3;
  const int thread_offset = threadIdx.x % 3;
  const int global_task_idx = blockIdx.x * task_num + task_offset;
  const bool is_control_thread = (thread_offset == 0);

  __shared__ mass_t mass[task_num];
  __shared__ int iterate_times[task_num];
  __shared__ int fail_iterate_times[task_num];

  __shared__ double center_step[task_num];
  __shared__ state_t y[task_num];
  __shared__ Eigen::Array33d acclerate_of_y[task_num];
  __shared__ double energy_of_y[task_num];
  __shared__ double time[task_num];

  __shared__ bool terminate[task_num];
  __shared__ int terminate_counter;

  // 30
  __shared__ double step[task_num][3];
  __shared__ state_t y_next[task_num][3];
  __shared__ Eigen::Array33d acclerate_of_y_next[task_num][3];
  __shared__ double energy_of_y_next[task_num][3];
  __shared__ bool is_ok[task_num][3];

  //__shared__ uint32_t will_go_on;

  // constexpr uint32_t terminate_flag = 0b1111111111;

  if (threadIdx.x == 0) {
    terminate_counter = 0;

    // will_go_on = 0;
  }

  // const uint32_t mask = (1ULL << task_offset);

  // initialize
  if (is_control_thread) {
    time[task_offset] = 0;
    mass[task_offset] = inputs[global_task_idx].mass;
    y[task_offset] = inputs[global_task_idx].beg_state;
    iterate_times[task_offset] = 0;
    fail_iterate_times[task_offset] = 0;
    center_step[task_offset] = opt.step_guess;

    ::libcudathreebody::compute_potential_acclerate(
        y[task_offset].position, mass[task_offset], energy_of_y + task_offset,
        acclerate_of_y + task_offset);

    energy_of_y[task_offset] +=
        compute_kinetic(y[task_offset].velocity, mass[task_offset]);

    terminate[task_offset] = false;
    // have_output[task_offset] = false;
  }

  __syncthreads();

  while (true) {
    // update will_go_on
    if (is_control_thread) {
      bool goon = true;

      if (center_step[task_offset] <= 1) {
        goon = false;
      }

      if (time[task_offset] >= opt.time_end) {
        goon = false;
      }

      bool add_counter = (goon == false) && (terminate[task_offset] == false);

      terminate[task_offset] = terminate[task_offset] || !goon;

      atomicAdd(&terminate_counter, int(add_counter));
    }

    __syncthreads();

    if (terminate_counter >= task_num) {
      break;
    }

    // execute by each thread
    {
      const double current_max_step = opt.time_end - time[task_offset];

      step[task_offset][thread_offset] =
          center_step[task_offset] / 2 * (1 << thread_offset);
      step[task_offset][thread_offset] =
          min(step[task_offset][thread_offset], current_max_step);
      is_ok[task_offset][thread_offset] = false;

      ::libcudathreebody::rk4_2(
          y[task_offset], mass[task_offset], step[task_offset][thread_offset],
          &y_next[task_offset][thread_offset], acclerate_of_y[task_offset]);
      ::libcudathreebody::compute_potential_acclerate(
          y_next[task_offset][thread_offset].position, mass[task_offset],
          &energy_of_y_next[task_offset][thread_offset],
          &acclerate_of_y_next[task_offset][thread_offset]);
      energy_of_y_next[task_offset][thread_offset] +=
          libcudathreebody::compute_kinetic(
              y_next[task_offset][thread_offset].velocity, mass[task_offset]);

      is_ok[task_offset][thread_offset] =
          abs((energy_of_y_next[task_offset][thread_offset] -
               energy_of_y[task_offset]) /
              energy_of_y[task_offset]) <= opt.max_relative_error;
    }

    __syncthreads();

    if (is_control_thread) {
      int accept_idx = -1;
      for (int idx = 0; idx < 3; idx++) {
        if (is_ok[task_offset][idx]) {
          accept_idx = idx;
        }
      }

      // const bool accept_result = (accept_idx >= 0) && (will_go_on & mask);
      const bool accept_result = (accept_idx >= 0) && (!terminate[task_offset]);

      if (accept_idx < 0) {
        center_step[task_offset] = step[task_offset][0] / 4;
        fail_iterate_times[task_offset]++;
      }

      if (accept_result) {
        center_step[task_offset] = step[task_offset][accept_idx];
        energy_of_y[task_offset] = energy_of_y_next[task_offset][accept_idx];
        energy_of_y[task_offset] = energy_of_y_next[task_offset][accept_idx];
        acclerate_of_y[task_offset] =
            acclerate_of_y_next[task_offset][accept_idx];
        y[task_offset] = y_next[task_offset][accept_idx];

        time[task_offset] += step[task_offset][accept_idx];

        iterate_times[task_offset]++;
      }
    }

    __syncthreads();
    //
  }

  __syncthreads();

  if (is_control_thread) {
    results[global_task_idx].end_time = time[task_offset];
    results[global_task_idx].end_energy = energy_of_y[task_offset];
    results[global_task_idx].end_state = y[task_offset];
    results[global_task_idx].fail_search_times =
        fail_iterate_times[task_offset];
    results[global_task_idx].iterate_times = iterate_times[task_offset];
  }
}
}  // namespace libcudathreebody

#endif  // THREEBODYFRACTAL3_LIBCUDATHREEBODY_INTERNAL_H