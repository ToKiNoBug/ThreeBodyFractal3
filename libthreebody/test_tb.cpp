#include <stdio.h>

#include <iostream>

#include "libthreebody.h"

using namespace libthreebody;

using std::cout, std::endl;

int main(int, char **) {
  input_t input;
  result_t result;

  input.mass = {3.20948, 1.84713, 4.6762};
  input.mass *= Ms / 20;

  input.beg_state.position = {{-1.03584, -0.0215062, 2.08068},
                              {-6.64071, 1.34016, -9.49566},
                              {-6.73013, 8.17534, 1.4536}};
  input.beg_state.position *= rs;
  input.beg_state.velocity = {{0.384347, 0.0969975, -0.50161},
                              {-0.697374, -0.766521, 0.250808},
                              {-0.394691, -0.192819, 0.747116}};
  input.beg_state.velocity *= vs;

  compute_options opt;
  opt.max_relative_error = 1e-4;
  opt.step_guess = 1e-2 * year;
  opt.time_end = 20 * year;

  cout << "original energy = " << compute_energy(input.beg_state, input.mass)
       << endl;

  simulate_2(input, opt, &result);

  cout << "result of position = \n" << result.end_state.position / rs << endl;
  cout << "result of velocity = \n" << result.end_state.velocity / vs << endl;

  cout << "result of energy = " << result.end_energy << endl;
  cout << "result of time = " << result.end_time / year << endl;

  cout << "iterate times = " << result.iterate_times << endl;
  cout << "fail iterate times = " << result.fail_search_times << endl;

  printf("Finished.\n");

  return 0;
}