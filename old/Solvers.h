#ifndef SOLVERS_H
#define SOLVERS_H

#include "Derivative.h"
#include "MultiRec.h"

int main();

template <int DimNum, int BodyNum>
class SolverBase {
 public:
  SolverBase() = default;
  ~SolverBase() = default;

  inline void setMass(BodyVec_t<BodyNum> &&absMass) noexcept {
    mass = absMass;
    computeSafeDistance<DimNum, BodyNum>(mass, &safeDistance);
  }

  inline void setMass(const BodyVec_t<BodyNum> &absMass) noexcept {
    mass = absMass;
    computeSafeDistance<DimNum, BodyNum>(mass, &safeDistance);
  }
  inline void setMaxYears(double yrs) noexcept { maxTime = yrs * year; }

  inline const BodyVec_t<BodyNum> &getMass() const noexcept { return mass; }

 protected:
  double maxTime;
  BodyVec_t<BodyNum> mass;
  ArrayBBd<BodyNum> safeDistance;
};

template <int DimNum, int BodyNum>
class ode45 : public SolverBase<DimNum, BodyNum> {
 public:
  ode45() = default;
  ~ode45() = default;

  double solve(const state_t<DimNum, BodyNum> &y0, double step,
               const double error) noexcept {
    static constexpr double searchRatio = 0.5;
    static constexpr int rank = 4;
    static const double ratio = std::pow(2, rank) - 1;

    MultiRec<state_t<DimNum, BodyNum>> recorder;
    recorder.push(0.0, y0);
    bool collide = false;
    while (true) {
      if (recorder.currentTime() > this->maxTime) {
        break;
      }
      const state_t<DimNum, BodyNum> &y = recorder.currentState();
      ArrayBBd<BodyNum> currentDistMat;
      computeDistance<DimNum, BodyNum>(y.position(), &currentDistMat);

      if ((currentDistMat < this->safeDistance).any()) {
        collide = true;
        // cout<<"Collided at "<<recorder.currentTime()/year<<" year(s)"<<endl;
        break;
      }

      const double minStep = 1e-6 * year;
      // 16*std::nextafter(recorder.currentTime(),recorder.currentTime()*2)-16*recorder.currentTime();

      const double curEnergy = y.energy(this->mass);

      state_t<DimNum, BodyNum> y_h;
      RK4(step, y, &y_h);

      // cout<<"step="<<step<<" , minStep="<<minStep<<endl;

      if (isErrorTolerantable(curEnergy, y_h.energy(this->mass), error)) {
        // current error is tolernable, scale up until is't not tolernable
        while (true) {
          step /= searchRatio;
          RK4(step, y, &y_h);
          if (!isErrorTolerantable(curEnergy, y_h.energy(this->mass), error)) {
            step *= searchRatio;
            RK4(step, y, &y_h);
            break;
          }
        }
      } else {
        // current error is not tolernable, scale down until tolernable
        while (true) {
          step *= searchRatio;
          RK4(step, y, &y_h);
          if (isErrorTolerantable(curEnergy, y_h.energy(this->mass), error)) {
            break;
          }
        }
      }

      if (step <= minStep) {
        collide = true;
        // cout<<"Predicted a collision at "<<recorder.currentTime()/year<<"
        // year(s)"<<endl;
        break;
      }

      recorder.push(recorder.currentTime() + step, y_h);
    }

    if (!collide) {
      return this->maxTime;
    } else {
      const state_t<DimNum, BodyNum> &y = recorder.currentState();
      ArrayBBd<BodyNum> matCur;  //,matPrev;
      computeDistance<DimNum, BodyNum>(y.position(), &matCur);
      // computeDistance<DimNum,BodyNum>(recorder.previousState().position(),&matPrev);

      // cout<<"MatPrev/rs=\n"<<matPrev/rs<<endl;
      /*
      cout<<"MatSafe/rs=\n"<<this->safeDistance/rs<<endl;
      cout<<"MatCur/rs=\n"<<matCur/rs<<endl;

      cout<<"MatCur/MatSafe=\n"<<matCur/this->safeDistance<<endl;

      */

      int i = 0, j = 0;

      (matCur / this->safeDistance).minCoeff(&i, &j);

      const double r2 = matCur(i, j);
      const double rs2 = this->safeDistance(i, j);

      const double dr2_div_dt =
          2 * ((y.position().col(i) - y.position().col(j)) *
               (y.velocity().col(i) - y.velocity().col(j)))
                  .sum();

      const double deltaT = (rs2 - r2) / dr2_div_dt;

      // cout<<"dr2/dt="<<dr2_div_dt<<" , "<<"deltaT="<<deltaT<<endl;

      return deltaT + recorder.currentTime();
    }
  }

 private:
  friend int main();
  inline void RK4(const double h, const state_t<DimNum, BodyNum> &y_n,
                  state_t<DimNum, BodyNum> *y_n1) const noexcept {
    state_t<DimNum, BodyNum> k1, k2, k3, k4;
    const double halfStep = h / 2;

    k1.position() = y_n.velocity();
    computeDiff<DimNum, BodyNum>(y_n.position(), this->mass, &k1.velocity());

    k2.position() = y_n.velocity() + halfStep * k1.velocity();
    computeDiff<DimNum, BodyNum>(y_n.position() + halfStep * k1.position(),
                                 this->mass, &k2.velocity());

    k3.position() = y_n.velocity() + halfStep * k2.velocity();
    computeDiff<DimNum, BodyNum>(y_n.position() + halfStep * k2.position(),
                                 this->mass, &k3.velocity());

    k4.position() = y_n.velocity() + h * k3.velocity();
    computeDiff<DimNum, BodyNum>(y_n.position() + h * k3.position(), this->mass,
                                 &k4.velocity());

    y_n1->vector() = y_n + h * (k1 + 2 / 6.0 * k2 + 2 / 6.0 * k3 + k4);
  }

  static inline bool isErrorTolerantable(const double curEnergy,
                                         const double nextEnergy,
                                         const double precision) noexcept {
    return std::abs(nextEnergy - curEnergy) <=
           (precision * std::abs(curEnergy));
  }
};

#endif  // SOLVERS_H
