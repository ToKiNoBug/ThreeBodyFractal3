#include "libcudathreebody.h"

#include <cuda.h>

void *allocate_device_memory(size_t bytes, int *errorcode = nullptr) noexcept {
  void *dptr = nullptr;
  cudaError_t ce = cudaMalloc(&dptr, bytes);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }

  return dptr;
}

void free_device_memory(void *device_ptr, int *errorcode = nullptr) noexcept {
  cudaError_t ce = cudaFree(&device_ptr);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }
}

void memcpy_host_to_device(const void *host_ptr, void *device_ptr, size_t bytes,
                           int *errorcode = nullptr) noexcept {
  cudaError_t ce =
      cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }
}

void memcpy_device_to_host(const void *device_ptr, void *host_ptr, size_t bytes,
                           int *errorcode = nullptr) noexcept {
  cudaError_t ce =
      cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }
}

void wait_for_device() noexcept { cudaDeviceSynchronize(); }