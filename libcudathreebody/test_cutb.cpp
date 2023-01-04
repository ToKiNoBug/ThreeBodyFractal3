#include <omp.h>
#include <stdio.h>

#include "libcudathreebody.h"
int main(int argC, char **argV) {
  {
    int error_code = 0;
    if (!libcudathreebody::is_device_ok(&error_code)) {
      printf("Fatal error : no avaliable cuda device. Error code : %i\n",
             error_code);
      return 1;
    }
  }
  using namespace libthreebody;
  const int num = 10000;
  input_t *const inputs = new input_t[num];
  result_t *const results = new result_t[num];

  void *const dev_inputs =
      libcudathreebody::allocate_device_memory(sizeof(input_t) * num);
  void *const dev_results =
      libcudathreebody::allocate_device_memory(sizeof(result_t) * num);

  if (dev_results == nullptr || dev_inputs == nullptr) {
    printf("\nError : failed to allocate device memory.\n");
    delete[] inputs;
    delete[] results;
    return 1;
  }

  compute_options opt;
  opt.max_relative_error = 1e-4;
  opt.step_guess = 1e-2 * year;
  opt.time_end = 20 * year;
  {
    const int period = 13;
    input_t center_input;

    center_input.mass = {3.20948, 1.84713, 4.6762};
    center_input.mass *= Ms / period;

    center_input.beg_state.position = {{-1.03584, -0.0215062, 2.08068},
                                       {-6.64071, 1.34016, -9.49566},
                                       {-6.73013, 8.17534, 1.4536}};
    center_input.beg_state.position *= rs;
    center_input.beg_state.velocity = {{0.384347, 0.0969975, -0.50161},
                                       {-0.697374, -0.766521, 0.250808},
                                       {-0.394691, -0.192819, 0.747116}};
    center_input.beg_state.velocity *= vs;

    for (int idx = 0; idx < num; idx++) {
      double p = (idx % period) + 1;
      inputs[idx].mass = center_input.mass * p;
      inputs[idx].beg_state = center_input.beg_state;
    }
  }
  double wtime = omp_get_wtime();
  libcudathreebody::run_cuda_simulations(inputs, results, dev_inputs,
                                         dev_results, num, opt);
  libcudathreebody::wait_for_device();
  wtime = omp_get_wtime() - wtime;

  printf("%i simulations finished in %F seconds. %F ms per simulation.\n", num,
         wtime, wtime / num * 1000);
  /*
  printf("Collide ages : [");
  for (int idx = 0; idx < num; idx++) {
    printf("%F, ", results[idx].end_time / year);
  }
  printf("];\n");
  */

  printf("iterate times and fail times : [");
  for (int idx = 0; idx < num; idx++) {
    printf("(%i, %i), ", results[idx].iterate_times,
           results[idx].fail_search_times);
  }
  printf("];\n");

  libcudathreebody::free_device_memory(dev_inputs);
  libcudathreebody::free_device_memory(dev_results);
  delete[] inputs;
  delete[] results;
  return 0;
}