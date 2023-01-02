#include "libcudathreebody.h"

#include <cuda.h>

#include "internal.h"

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

bool libcudathreebody::run_cuda_simulations(
    const libthreebody::input_t *const inputs_host,
    libthreebody::result_t *const dest_host, void *buffer_input_device,
    void *buffer_result_device, size_t num, libthreebody::compute_options &opt,
    int *errorcode) {
  cudaError_t ce;

  const int num_run_10 = 10 * ((num) / 10);

  printf("num_run_10 = %i\n", num_run_10);

  if (num_run_10 > 0) {
    ce = cudaMemcpy(buffer_input_device, inputs_host,
                    sizeof(input_t) * num_run_10, cudaMemcpyHostToDevice);
    if (ce != cudaError_t::cudaSuccess) {
      if (errorcode != nullptr) {
        *errorcode = ce;
      }
      return false;
    }

    libcudathreebody::simulate_10<<<num_run_10 / 10, 30>>>(
        (const input_t *)buffer_input_device, opt,
        (result_t *)buffer_result_device);
    printf("%i tasks added to gpu by %i blocks.\n", num_run_10,
           num_run_10 / 10);
  }

  for (int i = num_run_10; i < num; i++) {
    libthreebody::simulate_2(inputs_host[i], opt, &dest_host[i]);
  }

  if (num_run_10 > 0) {

    cudaDeviceSynchronize();

    ce = cudaMemcpy(dest_host, buffer_result_device,
                    sizeof(result_t) * num_run_10, cudaMemcpyDeviceToHost);
    printf("GPU finished %i tasks.\n", num_run_10);
  }

  if (ce != cudaError_t::cudaSuccess) {
    if (errorcode != nullptr) {
      *errorcode = ce;
    }
    return false;
  }
  return true;
}