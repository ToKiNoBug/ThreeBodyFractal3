#include <CLI11.hpp>
#include <atomic>
#include <fractal_utils/core_utils.h>
#include <iostream>
#include <libthreebodyfractal.h>
#include <mutex>
#include <omp.h>
#include <tbf-task.h>

#include <filesystem>

using std::cout, std::endl;

struct render_mem_resource {
  render_mem_resource() = delete;
  render_mem_resource(const render_mem_resource &) = delete;
  render_mem_resource(render_mem_resource &&) = delete;

  render_mem_resource(size_t rows, size_t cols);
  ~render_mem_resource();
  // fractal_utils::fractal_map map_result;
  const size_t capacity;
  void *const buffer;
};

class rsc_allocator {

private:
  // true means avaliable and false means allocated.
  std::map<const render_mem_resource *, bool> rsc_map;
  std::mutex lock;

public:
  const size_t rows;
  const size_t cols;
  rsc_allocator(size_t rows, size_t cols, int thread_num);
  ~rsc_allocator();

  const render_mem_resource *allocate() noexcept;
  void deallocate(const render_mem_resource *ptr) noexcept;
};

int main(int argc, char **argv) {

  std::string taskfile;

  CLI::App app;

  app.add_option("taskfile", taskfile, "Json task file to execute.")
      ->default_val("task.json")
      ->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  task_input task;

  if (!load_task_from_json(&task, taskfile)) {
    cout << "Failed to load task json from file " << taskfile << endl;
    return 1;
  }

  omp_set_num_threads(task.cpu_threads);

  int unfinished = 0;
  std::vector<uint8_t> lut_is_frame_finished;
  lut_is_frame_finished.resize(task.frame_count);

  for (uint8_t &val : lut_is_frame_finished) {
    val = false;
  }

  for (int fidx = 0; fidx < task.frame_count; fidx++) {
    std::string tbf = task.tbf_filename(fidx);
    if (!std::filesystem::exists(tbf)) {
      cout << "Error : tbf file " << tbf << " does not exist." << endl;
      return 1;
    }
    bool is_all_finished = true;
    for (int fpsidx = 0; fpsidx < task.fps; fpsidx++) {
      if (!std::filesystem::exists(task.png_filename(fidx, fpsidx))) {
        is_all_finished = false;
        break;
      }
    }
    if (!is_all_finished)
      unfinished++;

    lut_is_frame_finished[fidx] = is_all_finished;
  }

  if (unfinished <= 0) {
    cout << "All tasks finished." << endl;
    return 0;
  }

  size_t rows, cols;

  if (!libthreebody::load_fractal_basical_information_binary(task.center_source,
                                                             &rows, &cols)) {
    cout << "Failed to get rows and cols from file " << task.center_source
         << endl;
    return 1;
  }

  if (rows <= 2 || cols <= 2) {
    cout << "Error : rows = " << rows << ", cols = " << cols << endl;
    return 1;
  }

  rsc_allocator allocator(rows, cols, omp_get_num_threads());

  std::mutex lock_of_cout;
  std::atomic_int finished = 0;
#pragma omp parallel for schedule(dynamic)
  for (int fidx = 0; fidx < task.frame_count; fidx++) {
    if (lut_is_frame_finished[fidx]) {
      continue;
    }
    const std::string tbf_filename = task.tbf_filename(fidx);
    lock_of_cout.lock();
    cout << "\r[ " << finished << " / " << unfinished << " : "
         << float(finished) * 100 / unfinished << "% ] : rendering "
         << tbf_filename;
    lock_of_cout.unlock();

    const render_mem_resource *const resource = allocator.allocate();

    if (resource == nullptr) {
      cout << "Error : failed to allocate." << endl;
      exit(1);
    }

    for (int fpsidx = 0; fpsidx < task.fps; fpsidx++) {
      const std::string pngfilename = task.png_filename(fidx, fpsidx);
      if (std::filesystem::exists(pngfilename)) {
        continue;
      }

#warning Render the image here
    }

    allocator.deallocate(resource);
  }

  return 0;
}

render_mem_resource::render_mem_resource(size_t rows, size_t cols)
    : capacity(rows * cols * sizeof(libthreebody::result_t) * 2.5),
      buffer(fractal_utils::allocate_memory_aligned(32, this->capacity)) {
  if (buffer == nullptr) {
    cout << "Error : failed to allocate memory, buffer==nullptr" << endl;
    exit(1);
  }
  memset(this->buffer, 0, this->capacity);
}

render_mem_resource::~render_mem_resource() { free(this->buffer); }

rsc_allocator::rsc_allocator(size_t __rows, size_t __cols, int thread_num)
    : rows(__rows), cols(__cols) {
  for (int i = 0; i < thread_num; i++) {
    this->rsc_map[new render_mem_resource(__rows, __cols)] = true;
  }
}

rsc_allocator::~rsc_allocator() {
  for (auto &pair : this->rsc_map) {
    if (!pair.second) {
      cout << "Fatal error : the allocator is destroyed but a resource has not "
              "been freeed."
           << endl;
      exit(1);
    }

    delete pair.first;
  }

  this->rsc_map.clear();
}

const render_mem_resource *rsc_allocator::allocate() noexcept {
  std::lock_guard<std::mutex> lk(this->lock);

  for (auto &pair : this->rsc_map) {
    if (pair.second) {
      pair.second = false;
      return pair.first;
    }
  }

  render_mem_resource *const ptr =
      new render_mem_resource(this->rows, this->cols);

  this->rsc_map[ptr] = false;
  return ptr;
}
void rsc_allocator::deallocate(const render_mem_resource *ptr) noexcept {
  std::lock_guard<std::mutex> lk(this->lock);
  auto it = this->rsc_map.find(ptr);

  if (it == this->rsc_map.end()) {
    cout << "Fatal error : trying  to free a pointer that doesn not belong to "
            "a allocator."
         << endl;
    exit(1);
    return;
  }

  if (it->second == true) {
    cout << "Fatal error : double free" << endl;
    exit(1);
    return;
  }

  it->second = true;
}