#ifndef LIBTHREEBODY_LIBTHREEBODY_H
#define LIBTHREEBODY_LIBTHREEBODY_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <string_view>

#include "compile_time_pow.h"

#ifdef __CUDA_ARCH__
#define LIBTHREEBODY_HOST_DEVICE_FUN __device__ __host__
#else
#define LIBTHREEBODY_HOST_DEVICE_FUN
#endif

namespace libthreebody {

constexpr double G = 6.67259e-11;
constexpr double Ms = 2e30;
constexpr double year = 365 * 24 * 60 * 60;
constexpr double omega_s = 2 * M_PI / year;
constexpr double rho = 1.409e3;
constexpr double rs =
    0.5 * internal::inv_cubic(2 * G * Ms / (omega_s * omega_s));
constexpr double vs = omega_s * rs;
constexpr double as = omega_s * vs;

using mass_t = Eigen::Array3d;

struct alignas(32) state_t {
  Eigen::Array33d position;
  Eigen::Array33d velocity;

  LIBTHREEBODY_HOST_DEVICE_FUN inline double *data() noexcept {
    return this->position.data();
  }
  LIBTHREEBODY_HOST_DEVICE_FUN inline const double *data() const noexcept {
    return this->position.data();
  }

  LIBTHREEBODY_HOST_DEVICE_FUN constexpr inline int size() const noexcept {
    return 18;
  }
};

double compute_kinetic(const Eigen::Array33d &velocity,
                       const mass_t &mass) noexcept;
double compute_potential(const Eigen::Array33d &position,
                         const mass_t &mass) noexcept;
void compute_acclerate(const Eigen::Array33d &position, const mass_t &mass,
                       Eigen::Array33d *const acclerate) noexcept;
void compute_potential_acclerate(const Eigen::Array33d &position,
                                 const mass_t &mass, double *const potential,
                                 Eigen::Array33d *const acclerate) noexcept;

inline double compute_energy(const state_t &s, const mass_t &mass) noexcept {
  return compute_kinetic(s.velocity, mass) +
         compute_potential(s.position, mass);
}

void rk4(const state_t &y_n, const mass_t &mass, const double step,
         state_t *const y_n1) noexcept;

void rk4_2(const state_t &y_n, const mass_t mass, const double step,
           state_t *const y_n1,
           const Eigen::Array33d &acclerate_of_y_n) noexcept;

struct input_t {
  state_t beg_state;
  mass_t mass;
};

struct compute_options {
  double time_end;
  double max_relative_error;
  double step_guess;
};

struct result_t {
  state_t end_state;
  double end_time;
  double end_energy;
  int iterate_times;
  int fail_search_times;
};

void simulate(const input_t &input, const compute_options &opt,
              result_t *const result) noexcept;

void simulate_2(const input_t &input, const compute_options &opt,
                result_t *const result) noexcept;

bool load_parameters_from_D3B3(std::string_view filename,
                               mass_t *dest_mass = nullptr,
                               state_t *dest_begstate = nullptr,
                               compute_options *opt = nullptr) noexcept;

}  // namespace libthreebody

#endif  // LIBTHREEBODY_LIBTHREEBODY_H