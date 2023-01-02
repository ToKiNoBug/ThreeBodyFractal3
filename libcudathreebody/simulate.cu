#include <cmath>

#include "internal.h"

double __device__ libcudathreebody::compute_kinetic(
    const Eigen::Array33d &velocity, const mass_t &mass) noexcept {
  return ((velocity.square().colwise().sum()) * mass.transpose()).sum();
}

double __device__ libcudathreebody::compute_potential(
    const Eigen::Array33d &position, const mass_t &mass) noexcept {
  double pot = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      const double distance2 =
          (position.col(i) - position.col(j)).square().sum();
      const double invdistance = 1.0 / std::sqrt(distance2);
      pot -= G * mass(i) * mass(j) * invdistance;
    }
  }
  return pot;
}

void __device__ libcudathreebody::compute_acclerate(
    const Eigen::Array33d &x, const mass_t &mass,
    Eigen::Array33d *const dv) noexcept {

  dv->setZero();

  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      double temp = 0;
      Eigen::Array3d xj_sub_xi;

      for (int r = 0; r < 3; r++) {
        xj_sub_xi[r] = x(r, j) - x(r, i);
        temp += xj_sub_xi[r] * xj_sub_xi[r];
      }

      const double distanceSquare = temp;
      const double distance = std::sqrt(distanceSquare);

      for (int r = 0; r < 3; r++) {
        xj_sub_xi[r] *= G / (distanceSquare * distance);
        dv->operator()(r, i) += mass(j) * xj_sub_xi(r);
        dv->operator()(r, j) -= mass(i) * xj_sub_xi(r);
      }
    }
  }
}

void __device__ libcudathreebody::compute_potential_acclerate(
    const Eigen::Array33d &position, const mass_t &mass,
    double *const potential, Eigen::Array33d *const acclerate) noexcept {
  acclerate->setZero();
  double pot = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      Eigen::Array3d xj_sub_xi;

      double temp = 0;

      for (int r = 0; r < 3; r++) {
        xj_sub_xi[r] = position(r, j) - position(r, i);
        temp += xj_sub_xi[r] * xj_sub_xi[r];
      }

      // auto xj_sub_xi = x.col(j) - x.col(i);
      const double distanceSquare = temp;
      const double distance = std::sqrt(distanceSquare);

      for (int r = 0; r < 3; r++) {
        xj_sub_xi[r] *= G / (distanceSquare * distance);
        acclerate->operator()(r, i) += mass(j) * xj_sub_xi(r);
        acclerate->operator()(r, j) -= mass(i) * xj_sub_xi(r);
      }
      /*
      auto xj_sub_xi = position.col(j) - position.col(i);
      const double distanceSquare = xj_sub_xi.square().sum();
      const double distance = std::sqrt(distanceSquare);

      auto G_mult_diffXji_div_dist_pow_5_2 =
          G * xj_sub_xi / (distanceSquare * distance);

      acclerate->col(i) += mass(j) * G_mult_diffXji_div_dist_pow_5_2;
      acclerate->col(j) -= mass(i) * G_mult_diffXji_div_dist_pow_5_2;
      */

      pot -= G * mass(i) * mass(j) / distance;
    }
  }

  *potential = pot;
}

void __device__ rk4_update_state(double step, const double *y_nd,
                                 const double *k1d, const double *k2d,
                                 const double *k3d, const double *k4d,
                                 double *y_n1d) {
  for (int idx = 0; idx < 18; idx++) {
    y_n1d[idx] = y_nd[idx] +
                 step / 6 * (k1d[idx] + 2 * k2d[idx] + 2 * k3d[idx] + k4d[idx]);
  }
}

inline Eigen::Array33d __device__
rk4_update_position(const Eigen::Array33d &y_n_pos, const double __time_step,
                    const Eigen::Array33d &k_pos) noexcept {
  Eigen::Array33d ret;

  const double *const p_y_n_pos = y_n_pos.data();
  const double *const p_k_pos = k_pos.data();

  for (int idx = 0; idx < 9; idx++) {
    ret(idx) = y_n_pos(idx) + __time_step * k_pos(idx);
  }

  // ret(idx) = y_n_pos(idx) + __time_step * k_pos(idx);
  return ret;
}

void __device__ libcudathreebody::rk4(const state_t &y_n, const mass_t &mass,
                                      const double step,
                                      state_t *const y_n1) noexcept {
  const double half_step = step / 2;

  state_t k1, k2, k3, k4;

  k1.position = y_n.velocity;
  compute_acclerate(y_n.position, mass, &k1.velocity);

  k2.position = y_n.velocity + half_step * k1.velocity;
  compute_acclerate(y_n.position + half_step * k1.position, mass, &k2.velocity);

  k3.position = y_n.velocity + half_step * k2.velocity;
  compute_acclerate(y_n.position + half_step * k2.position, mass, &k3.velocity);

  k4.position = y_n.velocity + step * k3.velocity;
  compute_acclerate(y_n.position + step * k3.position, mass, &k4.velocity);

  const double *y_nd = y_n.data(), *k1d = k1.data(), *k2d = k2.data(),
               *k3d = k3.data(), *k4d = k4.data();

  double *y_n1d = y_n1->data();

  rk4_update_state(step, y_nd, k1d, k2d, k3d, k4d, y_n1d);
}

void __device__ libcudathreebody::rk4_2(
    const state_t &y_n, const mass_t mass, const double step,
    state_t *const y_n1, const Eigen::Array33d &acclerate_of_y_n) noexcept {
  const double half_step = step / 2;

  state_t k1, k2, k3, k4;

  k1.position = y_n.velocity;
  k1.velocity = acclerate_of_y_n;

  k2.position = y_n.velocity + half_step * k1.velocity;
  compute_acclerate(rk4_update_position(y_n.position, half_step, k1.position)
                    // y_n.position + half_step * k1.position
                    ,
                    mass, &k2.velocity);

  k3.position = y_n.velocity + half_step * k2.velocity;
  compute_acclerate(rk4_update_position(y_n.position, half_step, k2.position),
                    mass, &k3.velocity);

  k4.position = y_n.velocity + step * k3.velocity;
  compute_acclerate(rk4_update_position(y_n.position, step, k3.position), mass,
                    &k4.velocity);

  const double *y_nd = y_n.data(), *k1d = k1.data(), *k2d = k2.data(),
               *k3d = k3.data(), *k4d = k4.data();

  double *y_n1d = y_n1->data();

  rk4_update_state(step, y_nd, k1d, k2d, k3d, k4d, y_n1d);
}

void __global__ libcudathreebody::simulate_10(const input_t *const inputs,
                                              const compute_options opt,
                                              result_t *const results) {

  if (blockDim.x != 30) {
    return;
  }

  const int task_offset = threadIdx.x / 3;
  const int thread_offset = threadIdx.x % 3;
  const int global_task_idx = blockIdx.x * 10 + task_offset;
  const bool is_control_thread = (thread_offset == 0);

  __shared__ mass_t mass[10];
  __shared__ int iterate_times[10];
  __shared__ int fail_iterate_times[10];

  __shared__ double center_step[10];
  __shared__ state_t y[10];
  __shared__ Eigen::Array33d acclerate_of_y[10];
  __shared__ double energy_of_y[10];
  __shared__ double time[10];

  // 30
  __shared__ double step[10][3];
  __shared__ state_t y_next[10][3];
  __shared__ Eigen::Array33d acclerate_of_y_next[10][3];
  __shared__ double energy_of_y_next[10][3];
  __shared__ bool is_ok[10][3];

  __shared__ uint32_t will_go_on;

  constexpr uint32_t terminate_flag = 0b1111111111;

  if (threadIdx.x == 0) {
    will_go_on = 0;
  }

  const uint32_t mask = (1ULL << task_offset);

  // initialize
  if (is_control_thread) {
    time[task_offset] = 0;
    mass[task_offset] = inputs[global_task_idx].mass;
    y[task_offset] = inputs[global_task_idx].beg_state;
    iterate_times[task_offset] = 0;
    fail_iterate_times[task_offset] = 0;
    center_step[task_offset] = opt.step_guess;

    compute_potential_acclerate(y[task_offset].position, mass[task_offset],
                                energy_of_y + task_offset,
                                acclerate_of_y + task_offset);

    energy_of_y[task_offset] +=
        compute_kinetic(y[task_offset].velocity, mass[task_offset]);
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

      const uint32_t temp = (goon) ? (0UL) : (mask);

      atomicOr(&will_go_on, temp);
    }

    __syncthreads();
    if (will_go_on == terminate_flag) {
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

      libcudathreebody::rk4_2(
          y[task_offset], mass[task_offset], step[task_offset][thread_offset],
          &y_next[task_offset][thread_offset], acclerate_of_y[task_offset]);
      libcudathreebody::compute_potential_acclerate(
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

      const bool accept_result = (accept_idx >= 0) && (will_go_on & mask);

      if (accept_result < 0) {
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