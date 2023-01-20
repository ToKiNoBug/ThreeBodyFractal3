#include <CLI11.hpp>
#include <atomic>
#include <fractal_utils/core_utils.h>
#include <fractal_utils/png_utils.h>
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

  fractal_utils::fractal_map map_result;
};

class rsc_allocator {

private:
  // true means avaliable and false means allocated.
  std::map<render_mem_resource *, bool> rsc_map;
  std::mutex lock;

public:
  const size_t rows;
  const size_t cols;
  rsc_allocator(size_t rows, size_t cols, int thread_num);
  ~rsc_allocator();

  render_mem_resource *allocate() noexcept;
  void deallocate(render_mem_resource *ptr) noexcept;
};

int run_task(const task_input &task) noexcept;

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

  return run_task(task);
}

int run_task(const task_input &task) noexcept {

  libthreebody::color_map_all color_map = libthreebody::default_color_map_0;

  if (task.render_json != "default") {
    if (!libthreebody::load_color_map_all_from_file(task.render_json.data(),
                                                    &color_map)) {
      cout << "Failed to load color map " << task.render_json << endl;
      return 1;
    }
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
  libthreebody::compute_options opt;

  if (!libthreebody::load_fractal_basical_information_binary(
          task.center_source, &rows, &cols, nullptr, nullptr, &opt)) {
    cout << "Failed to get rows and cols from file " << task.center_source
         << endl;
    return 1;
  }

  if (rows <= 2 || cols <= 2) {
    cout << "Error : rows = " << rows << ", cols = " << cols << endl;
    return 1;
  }

  const double time_end = opt.time_end;

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

    render_mem_resource *const resource = allocator.allocate();

    if (resource == nullptr) {
      cout << "Error : failed to allocate." << endl;
      exit(1);
    }

    // load the binfile
    {
      fractal_utils::binfile binfile;

      if (!binfile.parse_from_file(tbf_filename.data())) {
        cout << "Failed to parse " << tbf_filename << endl;
        exit(1);
      }
      if (!libthreebody::fractal_bin_file_get_result(
              binfile, &resource->map_result, resource->buffer,
              resource->capacity)) {
        cout << "Failed to get result from binfile " << tbf_filename << endl;
        exit(1);
      }
    }

    fractal_utils::fractal_map map_u8c3(rows, cols, 3, resource->buffer);
    memset(map_u8c3.data, 0xFF, map_u8c3.byte_count());

    void *const new_buffer =
        (void *)((uint8_t *)resource->buffer + map_u8c3.byte_count());
    const size_t new_capacity = resource->capacity - map_u8c3.byte_count();

    std::vector<fractal_utils::pixel_RGB *> row_ptrs;

    row_ptrs.resize(rows);
    for (int r = 0; r < rows; r++) {
      row_ptrs[r] = nullptr;
    }

    for (int fpsidx = 0; fpsidx < (task.fps + task.extra_fps); fpsidx++) {
      const std::string pngfilename = task.png_filename(fidx, fpsidx);
      if (std::filesystem::exists(pngfilename)) {
        continue;
      }

      const double skip_ratio =
          1.0 - std::pow(task.zoom_speed, -double(fpsidx) / task.fps);

      const int skip_rows = std::floor(rows * skip_ratio / 2);
      const int skip_cols = std::floor(cols * skip_ratio / 2);

      // render the image
      if (!libthreebody::render_universial(resource->map_result,
                                           {skip_rows, skip_cols}, new_buffer,
                                           new_capacity, &map_u8c3, time_end)) {
        cout << "Failed to render " << tbf_filename << " at fpsidx = " << fpsidx
             << endl;
        exit(1);
      }
      // export the image

      const int img_rows = rows - 2 * skip_rows;
      const int img_cols = cols - 2 * skip_cols;
      row_ptrs.resize(img_rows);

      for (int ir = 0; ir < img_rows; ir++) {
        const int mr = ir + skip_rows;

        row_ptrs[ir] =
            map_u8c3.address<fractal_utils::pixel_RGB>(mr, skip_cols);
      }
      if (!fractal_utils::write_png(
              pngfilename.data(), fractal_utils::color_space::u8c3,
              (void **)row_ptrs.data(), img_rows, img_cols)) {
        cout << "Failed to write png " << pngfilename << endl;
        exit(1);
      }

      // a png is generated now.
    }
    // all pngs for this tbf file is generated.

    allocator.deallocate(resource);
  }

  return 0;
}

render_mem_resource::render_mem_resource(size_t rows, size_t cols)
    : capacity(rows * cols * sizeof(libthreebody::result_t) * 2.5),
      buffer(fractal_utils::allocate_memory_aligned(32, this->capacity)),
      map_result(rows, cols, sizeof(libthreebody::result_t)) {
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

render_mem_resource *rsc_allocator::allocate() noexcept {
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
void rsc_allocator::deallocate(render_mem_resource *ptr) noexcept {
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