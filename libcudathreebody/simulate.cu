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

      // xj_sub_xi[3] = 0;

      // auto xj_sub_xi = x.col(j) - x.col(i);
      const double distanceSquare = temp;
      const double distance = std::sqrt(distanceSquare);

      for (int r = 0; r < 3; r++) {
        xj_sub_xi[r] *= G / (distanceSquare * distance);
        dv->operator()(r, i) += mass(j) * xj_sub_xi(r);
        dv->operator()(r, j) -= mass(i) * xj_sub_xi(r);
      }

      /*
      auto G_mult_diffXji_div_dist_pow_5_2 =
          G * xj_sub_xi / (distanceSquare * distance);
      dv->col(i) += mass(j) * G_mult_diffXji_div_dist_pow_5_2;
      dv->col(j) -= mass(i) * G_mult_diffXji_div_dist_pow_5_2;
      */
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
    y_n1d[idx] = y_nd[idx] + step * (k1d[idx] + 1 / 3.0 * k2d[idx] +
                                     1 / 3.0 * k3d[idx] + k4d[idx]);
  }
}

void __device__ libcudathreebody::rk4(const state_t &y_n, const mass_t &mass,
                                      const double step,
                                      state_t *const y_n1) noexcept {}

void __device__ libcudathreebody::rk4_2(
    const state_t &y_n, const mass_t mass, const double step,
    state_t *const y_n1, const Eigen::Array33d &acclerate_of_y_n) noexcept {}