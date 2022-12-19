#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <math.h>

#include <Eigen/Dense>
#include <iostream>

using std::cout, std::cin, std::endl;

#define __dispLine cout << __FILE__ << " , " << __LINE__ << endl;

template <int BodyNum>
using BodyVec_t = Eigen::Array<double, BodyNum, 1>;

template <int BodyNum>
using ArrayBBd = Eigen::Array<double, BodyNum, BodyNum>;

template <int DimNum, int BodyNum>
using ArrayDBd = Eigen::Array<double, DimNum, BodyNum>;

constexpr double G = 6.67259e-11;
constexpr double Ms = 2e30;
constexpr double year = 365 * 24 * 60 * 60;
constexpr double omega_s = 2 * M_PI / year;
constexpr double rho = 1.409e3;
const double rs = 0.5 * std::pow(2 * G * Ms / (omega_s * omega_s), 1.0 / 3);
const double vs = omega_s * rs;
const double as = omega_s * vs;

template <int DimNum, int BodyNum>
class state_t : public Eigen::Array<double, DimNum * BodyNum * 2, 1> {
 public:
  state_t() { this->setZero(); }

  inline Eigen::Array<double, DimNum * BodyNum * 2, 1> &vector() {
    return *this;
  }

  inline ArrayDBd<DimNum, BodyNum> &position() {
    return *reinterpret_cast<ArrayDBd<DimNum, BodyNum> *>(this);
  }

  inline const ArrayDBd<DimNum, BodyNum> &position() const {
    return *reinterpret_cast<const ArrayDBd<DimNum, BodyNum> *>(this);
  }

  inline ArrayDBd<DimNum, BodyNum> &velocity() {
    return *(reinterpret_cast<ArrayDBd<DimNum, BodyNum> *>(this) + 1);
  }

  inline const ArrayDBd<DimNum, BodyNum> &velocity() const {
    return *(reinterpret_cast<const ArrayDBd<DimNum, BodyNum> *>(this) + 1);
  }

  inline double energy(const BodyVec_t<BodyNum> &mass) const {
    return kinetic(mass) + potential(mass);
  }

  inline double kinetic(const BodyVec_t<BodyNum> &mass) const {
    return ((velocity().square().colwise().sum()) * mass.transpose()).sum();
  }

  double potential(const BodyVec_t<BodyNum> &mass) const {
    double pot = 0;
    for (int32_t i = 0; i < 3; i++) {
      for (int32_t j = i + 1; j < 3; j++) {
        const double distance2 =
            (position().col(i) - position().col(j)).square().sum();
        const double invdistance = 1.0 / std::sqrt(distance2);
        pot -= G * mass(i) * mass(j) * invdistance;
      }
    }
    return pot;
  }
};

template <int DimNum, int BodyNum>
inline void computeDistance(const ArrayDBd<DimNum, BodyNum> &pos,
                            ArrayBBd<BodyNum> *dest) {
  dest->fill(std::numeric_limits<double>::infinity());
  for (int32_t i = 0; i < 3; i++) {
    for (int32_t j = i + 1; j < 3; j++) {
      const double distance =
          std::sqrt((pos.col(i) - pos.col(j)).square().sum());
      (*dest)(i, j) = distance;
      (*dest)(j, i) = distance;
    }
  }
}

template <int DimNum, int BodyNum>
inline void computeSafeDistance(const BodyVec_t<BodyNum> &mass,
                                ArrayBBd<BodyNum> *result) noexcept {
  // result->setZero();
  BodyVec_t<BodyNum> radius = (3 * mass / (4 * M_PI * rho)).pow(1.0 / 3);
  auto massMat = radius.replicate(1, 3);

  // dest=2.44*massMat.max(massMat.transpose());
  *result = 2.44 * massMat.max(massMat.transpose());
}
/*
template<int DimNum,int BodyNum>
bool computeDiff(const ArrayDBd<DimNum,BodyNum> &x,const BodyVec_t<BodyNum> &
mass, const ArrayDBd<DimNum,BodyNum> & safeDistance,ArrayDBd<DimNum,BodyNum> *
dv,ArrayDBd<DimNum,BodyNum> * unsafeDistanceDest);

*/

template <int DimNum, int BodyNum>
inline void computeDiff(const ArrayDBd<DimNum, BodyNum> &x,
                        const BodyVec_t<BodyNum> &mass,
                        ArrayDBd<DimNum, BodyNum> *dv) {
  dv->setZero();

  for (int32_t i = 0; i < 3; i++) {
    for (int32_t j = i + 1; j < 3; j++) {
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

#endif  // DERIVATIVE_H
