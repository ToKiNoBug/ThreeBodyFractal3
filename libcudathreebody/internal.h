#ifndef THREEBODYFRACTAL3_LIBCUDATHREEBODY_INTERNAL_H
#define THREEBODYFRACTAL3_LIBCUDATHREEBODY_INTERNAL_H

#include "../libthreebody/libthreebody.h"

namespace libcudathreebody {
constexpr double __constant__ G = ::libthreebody::G;
constexpr double __constant__ Ms = ::libthreebody::year;
constexpr double __constant__ omega_s = ::libthreebody::omega_s;
constexpr double rho = ::libthreebody::rho;
constexpr double rs = ::libthreebody::rs;
constexpr double vs = ::libthreebody::vs;
constexpr double as = ::libthreebody::as;

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

void __device__ rk4(const state_t &y_n, const mass_t &mass, const double step,
                    state_t *const y_n1) noexcept;

void __device__ rk4_2(const state_t &y_n, const mass_t mass, const double step,
                      state_t *const y_n1,
                      const Eigen::Array33d &acclerate_of_y_n) noexcept;

void __global__ simulate_10(const input_t *const inputs,
                            const compute_options opt, result_t *const results);

}  // namespace libcudathreebody

#endif  // THREEBODYFRACTAL3_LIBCUDATHREEBODY_INTERNAL_H