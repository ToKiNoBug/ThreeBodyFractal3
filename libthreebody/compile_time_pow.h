#ifndef THREEBODYFRACTAL3_COMPILE_TIME_POW_H
#define THREEBODYFRACTAL3_COMPILE_TIME_POW_H

namespace libthreebody {
namespace internal {

constexpr double abs(double x) {
  if (x >= 0) {
    return x;
  } else {
    return -x;
  }
}

constexpr double inv_cubic(double A) {
  double x = A;

  double prev_x = -x;

  while (true) {
    if (prev_x == x)
      break;
    prev_x = x;
    x -= (x * x * x - A) / (3 * x * x);
  }

  return x;
}
} // namespace internal
} // namespace libthreebody

#endif // THREEBODYFRACTAL3_COMPILE_TIME_POW_H