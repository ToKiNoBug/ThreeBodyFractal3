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
      const double distance = sqrt(distanceSquare);

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
      const double distance = sqrt(distanceSquare);

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

void __device__ libcudathreebody::rk4_update_state(
    double step, const double *y_nd, const double *k1d, const double *k2d,
    const double *k3d, const double *k4d, double *y_n1d) noexcept {
  step /= 6;
  for (int idx = 0; idx < 18; idx++) {
    const double temp =
        __fma_rn(2, k2d[idx], k1d[idx]) + __fma_rn(2, k3d[idx], k4d[idx]);
    y_n1d[idx] = __fma_rn(step, temp, y_nd[idx]);
  }
}

inline Eigen::Array33d __device__
rk4_update_position(const Eigen::Array33d &y_n_pos, const double __time_step,
                    const Eigen::Array33d &k_pos) noexcept {
  Eigen::Array33d ret;

  const double *const p_y_n_pos = y_n_pos.data();
  const double *const p_k_pos = k_pos.data();

  for (int idx = 0; idx < 9; idx++) {
    // ret(idx) = y_n_pos(idx) + __time_step * k_pos(idx);
    ret(idx) = __fma_rn(__time_step, k_pos(idx), y_n_pos(idx));
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

#include "libcudathreebody.h"

bool libcudathreebody::run_cuda_simulations(
    const libthreebody::input_t *const inputs_host,
    libthreebody::result_t *const dest_host, void *buffer_input_device,
    void *buffer_result_device, size_t num,
    const libthreebody::compute_options &opt, int *errorcode) noexcept {
  cudaError_t ce;

  constexpr int tasks_per_block = 21;

  const int num_run_gpu = tasks_per_block * ((num) / tasks_per_block);

  // printf("num_run_gpu = %i\n", num_run_gpu);

  if (num_run_gpu > 0) {
    ce = cudaMemcpy(buffer_input_device, inputs_host,
                    sizeof(input_t) * num_run_gpu, cudaMemcpyHostToDevice);
    if (ce != cudaError_t::cudaSuccess) {
      if (errorcode != nullptr) {
        *errorcode = ce;
      }
      return false;
    }

    libcudathreebody::simulate_N<tasks_per_block>
        <<<num_run_gpu / tasks_per_block, 3 * tasks_per_block>>>(
            (const input_t *)buffer_input_device, opt,
            (result_t *)buffer_result_device);
  }

  for (int i = num_run_gpu; i < num; i++) {
    libthreebody::simulate_2(inputs_host[i], opt, &dest_host[i]);
  }

  if (num_run_gpu > 0) {

    cudaDeviceSynchronize();

    ce = cudaMemcpy(dest_host, buffer_result_device,
                    sizeof(result_t) * num_run_gpu, cudaMemcpyDeviceToHost);
    // printf("GPU finished %i tasks.\n", num_run_gpu);
  }

  if (ce != cudaError_t::cudaSuccess) {
    if (errorcode != nullptr) {
      *errorcode = ce;
    }
    return false;
  }
  return true;
}