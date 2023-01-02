#ifndef THREEBODYFRACTAL3_LIBCUDATHREEBODY_H
#define THREEBODYFRACTAL3_LIBCUDATHREEBODY_H

#include "../libthreebody/libthreebody.h"

namespace libcudathreebody {

void *allocate_device_memory(size_t bytes, int *errorcode = nullptr) noexcept;

void free_device_memory(void *device_ptr, int *errorcode = nullptr) noexcept;

void memcpy_host_to_device(const void *host_ptr, void *device_ptr, size_t bytes,
                           int *errorcode = nullptr) noexcept;

void memcpy_device_to_host(const void *device_ptr, void *host_ptr, size_t bytes,
                           int *errorcode = nullptr) noexcept;

void wait_for_device() noexcept;

}  // namespace libcudathreebody

#endif  // THREEBODYFRACTAL3_LIBCUDATHREEBODY_H