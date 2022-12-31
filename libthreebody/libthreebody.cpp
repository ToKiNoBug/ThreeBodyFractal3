#include "libthreebody.h"

#include <immintrin.h>
#include <xmmintrin.h>

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
      double temp = 0;
      alignas(32) Eigen::Array4d xj_sub_xi;

      for (int r = 0; r < 3; r++) {
        xj_sub_xi[r] = x(r, j) - x(r, i);
        temp += xj_sub_xi[r] * xj_sub_xi[r];
      }

      xj_sub_xi[3] = 0;

      // auto xj_sub_xi = x.col(j) - x.col(i);
      const double distanceSquare = temp;
      const double distance = std::sqrt(distanceSquare);

      {
        __m256d j_sub_i = _mm256_load_pd(xj_sub_xi.data());
        __m256d mul = _mm256_set1_pd(G / (distanceSquare * distance));

        __m256d res = _mm256_mul_pd(j_sub_i, mul);
        _mm256_store_pd(xj_sub_xi.data(), res);
      }

      for (int r = 0; r < 3; r++) {
        // xj_sub_xi[r] *= G / (distanceSquare * distance);
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

void libthreebody::compute_potential_acclerate(
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

void rk4_update_state_fma(double step, const double *y_nd, const double *k1d,
                          const double *k2d, const double *k3d,
                          const double *k4d, double *y_n1d) {
  int idx;
  for (idx = 0; idx < 16; idx += 4) {
    const __m256d one_div_three = _mm256_set1_pd(1.0 / 3.0);
    const __m256d step_ = _mm256_set1_pd(step);

    __m256d temp;
    {
      __m256d k1_ = _mm256_load_pd(k1d + idx);
      __m256d k2_ = _mm256_load_pd(k2d + idx);
      __m256d k3_ = _mm256_load_pd(k3d + idx);
      __m256d k4_ = _mm256_load_pd(k4d + idx);

      __m256d temp1 = _mm256_fmadd_pd(k2_, one_div_three, k1_);
      __m256d temp2 = _mm256_fmadd_pd(k3_, one_div_three, k4_);

      temp = _mm256_add_pd(temp1, temp2);
    }

    __m256d res = _mm256_fmadd_pd(temp, step_, _mm256_load_pd(y_nd + idx));

    _mm256_store_pd(y_n1d + idx, res);
  }

  {
    const __m128d one_div_three = _mm_set1_pd(1.0 / 3.0);
    const __m128d step_ = _mm_set1_pd(step);

    __m128d temp;
    {
      __m128d k1_ = _mm_load_pd(k1d + idx);
      __m128d k2_ = _mm_load_pd(k2d + idx);
      __m128d k3_ = _mm_load_pd(k3d + idx);
      __m128d k4_ = _mm_load_pd(k4d + idx);

      __m128d temp1 = _mm_fmadd_pd(k2_, one_div_three, k1_);
      __m128d temp2 = _mm_fmadd_pd(k3_, one_div_three, k4_);

      temp = _mm_add_pd(temp1, temp2);
    }

    __m128d res = _mm_fmadd_pd(temp, step_, _mm_load_pd(y_nd + idx));

    _mm_store_pd(y_n1d + idx, res);
  }
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

inline Eigen::Array33d
rk4_update_position(const Eigen::Array33d &y_n_pos, const double __time_step,
                    const Eigen::Array33d &k_pos) noexcept {
  alignas(32) Eigen::Array33d ret;
  const __m256d time_step = _mm256_set1_pd(__time_step);

  const double *const p_y_n_pos = y_n_pos.data();
  const double *const p_k_pos = k_pos.data();
  int idx;
  for (idx = 0; idx < 8; idx += 4) {
    __m256d ynp = _mm256_load_pd(p_y_n_pos + idx);
    __m256d kp = _mm256_load_pd(p_k_pos + idx);
    __m256d res = _mm256_fmadd_pd(time_step, kp, ynp);
    _mm256_store_pd(ret.data() + idx, res);
  }

  ret(idx) = y_n_pos(idx) + __time_step * k_pos(idx);
  return ret;
}

void libthreebody::rk4_2(const state_t &y_n, const mass_t mass,
                         const double step, state_t *const y_n1,
                         const Eigen::Array33d &acclerate_of_y_n) noexcept {
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

  rk4_update_state_fma(step, y_nd, k1d, k2d, k3d, k4d, y_n1d);

  /*
    y_n1->position = y_n.position + step * (k1.position + 2 / 6.0 *
    k2.position
    + 2 / 6.0 * k3.position + k4.position); y_n1->velocity = y_n.velocity +
    step
    * (k1.velocity + 2 / 6.0 * k2.velocity + 2 / 6.0 * k3.velocity +
    k4.velocity);
                                            */
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

void libthreebody::simulate_2(const input_t &__i, const compute_options &opt,
                              result_t *const result) noexcept {
  const mass_t mass = __i.mass;

  state_t y = __i.beg_state;
  Eigen::Array33d acclerate_of_y;
  double energy_of_y;
  double time = 0;

  double center_step = opt.step_guess;

  constexpr int search_range_size = 3;
  double step[search_range_size];
  state_t y_next[search_range_size];
  Eigen::Array33d acclerate_of_y_next[search_range_size];
  double energy_of_y_next[search_range_size];
  bool is_ok[search_range_size];

  int iterate_times = 0;
  int fail_search_times = 0;

  compute_potential_acclerate(y.position, mass, &energy_of_y, &acclerate_of_y);
  energy_of_y += compute_kinetic(y.velocity, mass);

  while (true) {
    if (center_step <= 1) {
      break;
    }

    // printf("energy of y = %F\n", energy_of_y);

    const double current_max_step = opt.time_end - time;

    // assert(current_max_step > 0);

    step[0] = center_step / 2;
    step[1] = center_step;
    step[2] = center_step * 2;
    for (int idx = 0; idx < 3; idx++) {
      step[idx] = std::min(step[idx], current_max_step);
      is_ok[idx] = false;
    }

    int accept_idx = -1;

    // try center_step
    {
      int try_idx = 1;
      rk4_2(y, mass, step[try_idx], y_next + try_idx, acclerate_of_y);
      compute_potential_acclerate(y_next[try_idx].position, mass,
                                  energy_of_y_next + try_idx,
                                  acclerate_of_y_next + try_idx);
      energy_of_y_next[try_idx] +=
          compute_kinetic(y_next[try_idx].velocity, mass);
      is_ok[try_idx] = std::abs((energy_of_y_next[try_idx] - energy_of_y) /
                                energy_of_y) <= opt.max_relative_error;
    }

    if (is_ok[1]) {
      // if acceptable, try [2]
      accept_idx = 1;
      {
        int try_idx = 2;
        rk4_2(y, mass, step[try_idx], y_next + try_idx, acclerate_of_y);
        compute_potential_acclerate(y_next[try_idx].position, mass,
                                    energy_of_y_next + try_idx,
                                    acclerate_of_y_next + try_idx);
        energy_of_y_next[try_idx] +=
            compute_kinetic(y_next[try_idx].velocity, mass);
        is_ok[try_idx] = std::abs((energy_of_y_next[try_idx] - energy_of_y) /
                                  energy_of_y) <= opt.max_relative_error;
      }

      if (is_ok[2]) {
        accept_idx = 2;
      }

    } else {
      // if not acceptable, try [1]

      {
        int try_idx = 0;
        rk4_2(y, mass, step[try_idx], y_next + try_idx, acclerate_of_y);
        compute_potential_acclerate(y_next[try_idx].position, mass,
                                    energy_of_y_next + try_idx,
                                    acclerate_of_y_next + try_idx);
        energy_of_y_next[try_idx] +=
            compute_kinetic(y_next[try_idx].velocity, mass);
        is_ok[try_idx] = std::abs((energy_of_y_next[try_idx] - energy_of_y) /
                                  energy_of_y) <= opt.max_relative_error;
      }

      if (is_ok[0]) {
        accept_idx = 0;
      }
    }
    /*
        printf("center_step = %F, is_ok = [%i, %i, %i], accept_idx = %i.\n",
               center_step, int(is_ok[0]), int(is_ok[1]), int(is_ok[2]),
               accept_idx);
               */

    if (accept_idx < 0) {
      center_step = step[0] / 4;
      fail_search_times++;
      continue;
    }

    center_step = step[accept_idx];

    // acceptable

    energy_of_y = energy_of_y_next[accept_idx];
    acclerate_of_y = acclerate_of_y_next[accept_idx];
    y = y_next[accept_idx];

    time += step[accept_idx];

    iterate_times++;

    if (time >= opt.time_end) {
      break;
    }
  }

  // end loop

  assert(time <= opt.time_end);

  result->end_time = time;
  result->end_energy = energy_of_y;
  result->end_state = y;
  result->fail_search_times = fail_search_times;
  result->iterate_times = iterate_times;
}

bool libthreebody::load_parameters_from_D3B3(std::string_view filename,
                                             mass_t *mass, state_t *begstate,
                                             compute_options *opt) noexcept {

  FILE *const fp = fopen(filename.data(), "rb");
  if (fp == nullptr) {
    printf("\nError : failed to open file %s\n", filename.data());
    return false;
  }

  std::array<uint32_t, 2> DC_BC;
  fread(DC_BC.data(), sizeof(uint32_t), 2, fp);

  if (DC_BC[0] != 3 || DC_BC[1] != 3) {
    printf("\nError : the file is not for 3 dim 3 bodies simulation, but %u "
           "dim %u bodies.\n",
           DC_BC[0], DC_BC[1]);
    return false;
  }

  if (mass != nullptr) {
    fread(mass->data(), sizeof(double), 3, fp);
  } else {
    fseek(fp, sizeof(double[3]), SEEK_CUR);
  }

  if (begstate != nullptr) {
    fread(begstate->position.data(), sizeof(double), 9, fp);
    fread(begstate->velocity.data(), sizeof(double), 9, fp);
  } else {
    fseek(fp, sizeof(double[18]), SEEK_CUR);
  }

  if (opt != nullptr) {
    double buffer[3];
    fread(buffer, sizeof(double), 3, fp);
    opt->max_relative_error = 1e-4;
    opt->step_guess = buffer[1];
    opt->time_end = buffer[2] - buffer[0];
  }

  fclose(fp);

  return true;
  /**

      {
          uint32_t dC=DIM_COUNT,bC=BODY_COUNT;
          file.write((const char*)&dC,sizeof(dC));
          file.write((const char*)&bC,sizeof(bC));
          //write dim_count, then body_count in uint32
      }

      //write mass
      file.write((const char*)mass.data(),sizeof(mass));

      //write position in row major
      file.write((const char*)y0.first.data(),sizeof(double)*y0.first.size());

      //write velocity in row major
      file.write((const char*)y0.second.data(),sizeof(double)*y0.second.size());

      file.write((const char*)&ts.first,sizeof(ts.first));
      file.write((const char*)&step,sizeof(step));
      file.write((const char*)&ts.second,sizeof(ts.second));
  */
}