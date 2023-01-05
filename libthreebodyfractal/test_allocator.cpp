#include <stdio.h>

#include <vector>

#include "memory_resource.h"
int main() {
  libthreebody::gpu_mem_allocator alloc(4, 2048);

  std::vector<const libthreebody::gpu_memory_resource_t *> ress;

  ress.resize(60);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < ress.size(); i++) {
    ress[i] = alloc.allocate();
    if (ress[i] == nullptr) {
      printf("idx = %i : failed.\n", i);
    } else {
      printf("idx = %i : success.\n", i);
    }
  }

  for (auto &i : ress) {
    if (i != nullptr) {
      alloc.deallocate(i);
      i = nullptr;
    }
  }

  return 0;
}