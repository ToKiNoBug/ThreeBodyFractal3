#ifndef THREEBODYFRACTAL3_LIBCUDATHREEBODY_H
#define THREEBODYFRACTAL3_LIBCUDATHREEBODY_H

#include "../libthreebody/libthreebody.h"

namespace libcudathreebody {

bool is_device_ok(int *errorcode = nullptr) noexcept;

void *allocate_device_memory(size_t bytes, int *errorcode = nullptr) noexcept;

bool free_device_memory(void *device_ptr, int *errorcode = nullptr) noexcept;

bool memcpy_host_to_device(const void *host_ptr, void *device_ptr, size_t bytes,
                           int *errorcode = nullptr) noexcept;

bool memcpy_device_to_host(const void *device_ptr, void *host_ptr, size_t bytes,
                           int *errorcode = nullptr) noexcept;

bool wait_for_device(int *errorcode = nullptr) noexcept;

bool run_cuda_simulations(const libthreebody::input_t *const inputs_host,
                          libthreebody::result_t *const dest_host,
                          void *buffer_input_device, void *buffer_result_device,
                          size_t num, const libthreebody::compute_options &opt,
                          int *errorcode = nullptr) noexcept;

}  // namespace libcudathreebody

#endif  // THREEBODYFRACTAL3_LIBCUDATHREEBODY_H