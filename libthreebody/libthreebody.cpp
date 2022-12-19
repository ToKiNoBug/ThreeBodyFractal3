#include "libthreebody.h"

using namespace libthreebody;

double libthreebody::compute_potential(const Eigen::Array33d &position,
                                       const mass_t &mass) noexcept {
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

double libthreebody::compute_kinetic(const Eigen::Array33d &velocity,
                                     const mass_t &mass) noexcept {
  return ((velocity.square().colwise().sum()) * mass.transpose()).sum();
}

void libthreebody::compute_acclerate(const Eigen::Array33d &x,
                                     const mass_t &mass,
                                     Eigen::Array33d *const dv) noexcept {
  dv->setZero();

  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      auto xj_sub_xi = x.col(j) - x.col(i);
      const double distanceSquare = xj_sub_xi.square().sum();
      const double distance = std::sqrt(distanceSquare);

      auto G_mult_diffXji_div_dist_pow_5_2 =
          G * xj_sub_xi / (distanceSquare * distance);

      dv->col(i) += mass(j) * G_mult_diffXji_div_dist_pow_5_2;
      dv->col(j) -= mass(i) * G_mult_diffXji_div_dist_pow_5_2;
    }
  }
}

void libthreebody::compute_potential_acclerate(
    const Eigen::Array33d &position, const mass_t &mass,
    double *const potential, Eigen::Array33d *const acclerate) noexcept {
  acclerate->setZero();
  double pot = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      auto xj_sub_xi = position.col(j) - position.col(i);
      const double distanceSquare = xj_sub_xi.square().sum();
      const double distance = std::sqrt(distanceSquare);

      auto G_mult_diffXji_div_dist_pow_5_2 =
          G * xj_sub_xi / (distanceSquare * distance);

      acclerate->col(i) += mass(j) * G_mult_diffXji_div_dist_pow_5_2;
      acclerate->col(j) -= mass(i) * G_mult_diffXji_div_dist_pow_5_2;

      pot -= G * mass(i) * mass(j) / distance;
    }
  }

  *potential = pot;
}

void libthreebody::rk4(const state_t &y_n, const mass_t &mass,
                       const double step, state_t *const y_n1) noexcept {
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

  y_n1->position = y_n.position + step * (k1.position + 2 / 6.0 * k2.position +
                                          2 / 6.0 * k3.position + k4.position);
  y_n1->velocity = y_n.velocity + step * (k1.velocity + 2 / 6.0 * k2.velocity +
                                          2 / 6.0 * k3.velocity + k4.velocity);
}

void libthreebody::rk4_2(const state_t &y_n, const mass_t &mass,
                         const double step, state_t *const y_n1,
                         const Eigen::Array33d &acclerate_of_y_n) noexcept {
  const double half_step = step / 2;

  state_t k1, k2, k3, k4;

  k1.position = y_n.velocity;
  k1.velocity = acclerate_of_y_n;

  k2.position = y_n.velocity + half_step * k1.velocity;
  compute_acclerate(y_n.position + half_step * k1.position, mass, &k2.velocity);

  k3.position = y_n.velocity + half_step * k2.velocity;
  compute_acclerate(y_n.position + half_step * k2.position, mass, &k3.velocity);

  k4.position = y_n.velocity + step * k3.velocity;
  compute_acclerate(y_n.position + step * k3.position, mass, &k4.velocity);

  y_n1->position = y_n.position + step * (k1.position + 2 / 6.0 * k2.position +
                                          2 / 6.0 * k3.position + k4.position);
  y_n1->velocity = y_n.velocity + step * (k1.velocity + 2 / 6.0 * k2.velocity +
                                          2 / 6.0 * k3.velocity + k4.velocity);
}

void libthreebody::simulate(const input_t &__i, const compute_options &opt,
                            result_t *const result) noexcept {
  const mass_t mass = __i.mass;

  state_t y = __i.beg_state;
  Eigen::Array33d acclerate_of_y;
  double energy_of_y;

  double step = opt.step_guess;

  double time = 0;

  state_t y_next;
  Eigen::Array33d acclerate_of_y_next;
  double energy_of_y_next;

  int iterate_times = 0;
  int fail_search_times = 0;

  double prev_success_step = 0;

  compute_potential_acclerate(y.position, mass, &energy_of_y, &acclerate_of_y);
  energy_of_y += compute_kinetic(y.velocity, mass);

  while (true) {

    const double current_max_step = opt.time_end - time;

    assert(current_max_step > 0);

    step = std::min(step, current_max_step);

    rk4_2(y, mass, step, &y_next, acclerate_of_y);
    compute_potential_acclerate(y_next.position, mass, &energy_of_y_next,
                                &acclerate_of_y_next);
    energy_of_y_next += compute_kinetic(y_next.velocity, mass);

    const bool is_ok = std::abs((energy_of_y_next - energy_of_y) /
                                energy_of_y) <= opt.max_relative_error;

    const double current_step = step;

    if (is_ok) {
      energy_of_y = energy_of_y_next;
      acclerate_of_y = acclerate_of_y_next;
      y = y_next;

      time += step;

      iterate_times++;

      if (prev_success_step == step) {
        step *= 2;
      } else {
        prev_success_step = step;
      }

    } else {
      fail_search_times++;

      step /= 2;
    }

    if (current_step <= 1) {
      break;
    }

    if (time >= opt.time_end) {
      break;
    }
  }

  assert(time <= opt.time_end);

  result->end_time = time;
  result->end_energy = energy_of_y;
  result->end_state = y;
  result->fail_search_times = fail_search_times;
  result->iterate_times = iterate_times;
}