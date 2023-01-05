#ifndef THREEBODYFRACTAL3_LIBTHREEBODYFRACTAL_COMPUTE_POOL
#define THREEBODYFRACTAL3_LIBTHREEBODYFRACTAL_COMPUTE_POOL

#include <atomic>
#include <map>

namespace libthreebody {

struct gpu_memory_resource_t {
  gpu_memory_resource_t(int num);
  gpu_memory_resource_t(gpu_memory_resource_t &&);

  gpu_memory_resource_t() = delete;
  gpu_memory_resource_t(const gpu_memory_resource_t &) = delete;

  ~gpu_memory_resource_t();
  void *host_input_buffer{nullptr};
  void *device_mem_input{nullptr};
  void *device_mem_result{nullptr};

  void release() noexcept;
};

class gpu_mem_allocator {
 public:
  gpu_mem_allocator() = delete;
  gpu_mem_allocator(const gpu_mem_allocator &) = delete;
  gpu_mem_allocator(gpu_mem_allocator &&) = delete;

  gpu_mem_allocator(int gpu_resource_count, int cols);

  ~gpu_mem_allocator();

  const gpu_memory_resource_t *allocate() noexcept;
  void deallocate(const gpu_memory_resource_t *res) noexcept;

  inline int size() const noexcept { return this->resources.size(); }

 private:
  // true means avaliable
  std::map<gpu_memory_resource_t *, std::atomic_bool> resources;
};

}  // namespace libthreebody

#endif  // THREEBODYFRACTAL3_LIBTHREEBODYFRACTAL_COMPUTE_POOL