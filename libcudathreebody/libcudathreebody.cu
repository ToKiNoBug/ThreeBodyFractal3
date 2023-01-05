#include "libcudathreebody.h"

#include <cuda.h>

bool libcudathreebody::is_device_ok(int *errorcode) noexcept {
  int num = 0;
  cudaError_t ce = cudaGetDeviceCount(&num);

  if (errorcode != nullptr) {
    *errorcode = ce;
  }

  if (num <= 0 || ce != cudaError_t::cudaSuccess) {
    return false;
  }

  return true;
}

void *libcudathreebody::allocate_device_memory(size_t bytes,
                                               int *errorcode) noexcept {
  void *dptr = nullptr;
  cudaError_t ce = cudaMalloc(&dptr, bytes);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }

  return dptr;
}

bool libcudathreebody::free_device_memory(void *device_ptr,
                                          int *errorcode) noexcept {
  cudaError_t ce = cudaFree(&device_ptr);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }
  if (ce == cudaError_t::cudaSuccess) {
    return false;
  }
  return true;
}

bool libcudathreebody::memcpy_host_to_device(const void *host_ptr,
                                             void *device_ptr, size_t bytes,
                                             int *errorcode) noexcept {
  cudaError_t ce =
      cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }
  if (ce == cudaError_t::cudaSuccess) {
    return false;
  }
  return true;
}

bool libcudathreebody::memcpy_device_to_host(const void *device_ptr,
                                             void *host_ptr, size_t bytes,
                                             int *errorcode) noexcept {
  cudaError_t ce =
      cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost);
  if (errorcode != nullptr) {
    *errorcode = ce;
  }
  if (ce == cudaError_t::cudaSuccess) {
    return false;
  }
  return true;
}

bool libcudathreebody::wait_for_device(int *errorcode) noexcept {
  cudaError_t ce = cudaDeviceSynchronize();
  if (errorcode != nullptr) {
    *errorcode = ce;
  }

  if (ce == cudaError_t::cudaSuccess) {
    return false;
  }
  return true;
}
