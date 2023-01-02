#ifndef THREEBODYFRACTAL3_COMPILE_TIME_POW_H
#define THREEBODYFRACTAL3_COMPILE_TIME_POW_H
#include <cmath>
namespace libthreebody {
namespace internal {
constexpr double inv_cubic(double A) {
  double x = A;

  double prev_x = x * -1;

  while (true) {
    if (std::abs((prev_x - x) / x) < 1e-60) break;
    prev_x = x;
    x -= (x * x * x - A) / (3 * x * x);
  }

  return x;
}
}  // namespace internal
}  // namespace libthreebody

#endif  // THREEBODYFRACTAL3_COMPILE_TIME_POW_H