#include "memory_resource.h"

#include <exception>

#include "../libcudathreebody/libcudathreebody.h"

libthreebody::gpu_memory_resource_t::gpu_memory_resource_t(int num) {
  this->host_input_buffer = malloc(sizeof(input_t) * num);
  if (host_input_buffer == nullptr) {
    printf("\nError : malloc failed.\n");
    exit(1);
  }

  int error_code = 0;

  this->device_mem_input = libcudathreebody::allocate_device_memory(
      sizeof(input_t) * num, &error_code);
  if (this->device_mem_input == nullptr) {
    printf("\nError : cuda malloc failed with error code %i.\n", error_code);
    free(this->host_input_buffer);
    exit(1);
  }

  this->device_mem_result = libcudathreebody::allocate_device_memory(
      sizeof(result_t) * num, &error_code);
  if (this->device_mem_result == nullptr) {
    printf("\nError : cuda malloc failed with error code %i.\n", error_code);
    free(this->host_input_buffer);
    libcudathreebody::free_device_memory(this->device_mem_input);
    exit(1);
  }
}

libthreebody::gpu_memory_resource_t::gpu_memory_resource_t(
    gpu_memory_resource_t &&src) {
  this->release();

  this->host_input_buffer = src.host_input_buffer;
  src.host_input_buffer = nullptr;

  this->device_mem_input = src.device_mem_input;
  src.device_mem_input = nullptr;

  this->device_mem_result = src.device_mem_result;
  src.device_mem_result = nullptr;
}

libthreebody::gpu_memory_resource_t::~gpu_memory_resource_t() {
  this->release();
}

void libthreebody::gpu_memory_resource_t::release() noexcept {
  if (this->host_input_buffer != nullptr) {
    free(this->host_input_buffer);
    this->host_input_buffer = nullptr;
  }
  if (this->device_mem_input != nullptr) {
    libcudathreebody::free_device_memory(this->device_mem_input);
    this->device_mem_input = nullptr;
  }

  if (this->device_mem_result != nullptr) {
    libcudathreebody::free_device_memory(this->device_mem_result);
    this->device_mem_result = nullptr;
  }
}

libthreebody::gpu_mem_allocator::gpu_mem_allocator(int gpu_resource_count,
                                                   int cols) {
  for (int i = 0; i < gpu_resource_count; i++) {
    resources[new gpu_memory_resource_t(cols)] = true;
  }
}

libthreebody::gpu_mem_allocator::~gpu_mem_allocator() {
  for (auto &r : this->resources) {
    if (!r.second) {
      printf("\nError : GPU resource allocated but not released after use.\n");
      exit(1);
      return;
    }

    delete r.first;
  }

  this->resources.clear();
}

const libthreebody::gpu_memory_resource_t *
libthreebody::gpu_mem_allocator::allocate() noexcept {
  // const libthreebody::gpu_memory_resource_t *ret = nullptr;
  for (auto &r : this->resources) {
    bool desired = true;

    const bool cas_success = r.second.compare_exchange_strong(desired, false);

    if (cas_success) {
      return r.first;
    }
  }

  return nullptr;
}

void libthreebody::gpu_mem_allocator::deallocate(
    const gpu_memory_resource_t *res) noexcept {
  auto it = this->resources.find(const_cast<gpu_memory_resource_t *>(res));

  if (it == this->resources.end()) {
    printf(
        "\nError : gpu resource failed to release : no such "
        "resource.\n");
    exit(1);
    return;
  }

  it->second = true;
}