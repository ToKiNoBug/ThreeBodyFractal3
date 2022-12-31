#include "libthreebody.h"

#include <iostream>
#include <stdio.h>

int main(int argC, char **argV) {
  using namespace libthreebody;
  input_t i;
  compute_options opt;

  bool ok = load_parameters_from_D3B3("select/02.paraD3B3", &i.mass,
                                      &i.beg_state, &opt);
  if (!ok) {
    printf("Failed.\n");
    return 1;
  }

  std::cout << "The mass is : [" << i.mass.transpose() / Ms << "]\n";
  std::cout << "The position is :\n[" << i.beg_state.position / rs << "]\n";
  std::cout << "The velocity is :\n[" << i.beg_state.velocity / vs << "]\n";

  std::cout << "The step is : " << opt.step_guess / year << '\n';
  std::cout << "The time_end is : " << opt.time_end / year << '\n';
  std::cout << std::endl;

  return 0;
}