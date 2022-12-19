#include <omp.h>

#include <iostream>

#include "Derivative.h"
#include "Solvers.h"

const state_t<2, 3> standardY0;

void initializeStdY0() {
  const Eigen::Array<double, 12, 1> stdY0Val(
      {-2.27203257188855, -0.519959453298081, 1.09628120907693,
       -1.98504043661515, 1.17575136281162, 2.50499988991323,
       -0.126186392581979, -0.262340533926015, -0.0512572533808924,
       -0.0609798828559385, 0.0762336331145880, 0.128100099879297});

  state_t<2, 3>& NCRef = const_cast<state_t<2, 3>&>(standardY0);
  NCRef.topRows(6) = stdY0Val.topRows(6) * rs;
  NCRef.bottomRows(6) = stdY0Val.bottomRows(6) * vs;
}

int main() {
  initializeStdY0();

  state_t<2, 3> y = standardY0;

  ode45<2, 3> sol;
  sol.setMass(BodyVec_t<3>({1, 2, 3}) * Ms);
  sol.setMaxYears(10);

  constexpr int loopN = 50;
  double clk = omp_get_wtime();
  for (int i = 0; i < loopN; i++)
    const double lifeTime = sol.solve(y, 1e-3 * year, 5e-3);
  clk = omp_get_wtime() - clk;

  cout << "Time cost=" << clk / loopN * 1e3 << "ms per loop" << endl;

  // cout<<"lifeTime/year="<<lifeTime/year<<endl;

  return 0;
}
