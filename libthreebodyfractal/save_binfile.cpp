#include <lzma.h>
#include <stdio.h>

#include <map>
// #define NBT_MAP_TYPE std::map
#include <nbt.hpp>

#include "threebodyfractal.h"

bool xz_compress(uint8_t *const dest, const uint8_t *const src,
                 const uint64_t src_bytes, const uint64_t dest_capacity,
                 uint64_t *const dest_bytes) noexcept;

bool fractal_bin_file_write_basical_information(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const libthreebody::input_t &center_input,
    const fractal_utils::center_wind<double> &wind,
    const libthreebody::compute_options &opt,
    const fractal_utils::fractal_map &mat_result) noexcept;

bool fractal_bin_file_write_end_state(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept;

bool fractal_bin_file_write_collide_time(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept;

bool fractal_bin_file_write_end_energy(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept;

bool fractal_bin_file_write_iterate_time(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept;

bool fractal_bin_file_write_fail_search_time(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept;

//
bool libthreebody::save_fractal_bin_file(
    std::string_view filename, const input_t &center_input,
    const fractal_utils::center_wind<double> &wind, const compute_options &opt,
    const fractal_utils::fractal_map &mat_result, void *const buffer,
    const size_t buffer_bytes) noexcept {
  using namespace fractal_utils;

  if (buffer_bytes < mat_result.byte_count() * 2) {
    printf("\nError : not enough buffer. The buffer should be at least 2 time "
           "bigger than the mat_result.\n");
    return false;
  }

  if (size_t(buffer) % 32 != 0) {
    printf("\nError : buffer not aligned as 32 bytes.\n");
    return false;
  }

  FILE *const fp = fopen(filename.data(), "wb");

  if (fp == NULL) {
    printf("\nFailed to create file %s.\n", filename.data());
    fclose(fp);
    return false;
  }

  // write file header
  {
    file_header header;

    fwrite(&header, 1, sizeof(header), fp);
  }

  uint8_t *const buffer_A = reinterpret_cast<uint8_t *>(buffer);
  uint8_t *const buffer_B =
      buffer_A + 32 * size_t(std::ceil(mat_result.byte_count() / 32.0f));

  const size_t buffer_A_capacity = buffer_B - buffer_A;
  const size_t buffer_B_capacity = buffer_A + buffer_bytes - buffer_B;

  printf("Writting basical information\n");
  if (!fractal_bin_file_write_basical_information(
          fp, buffer_A, buffer_A_capacity, buffer_B, buffer_B_capacity,
          center_input, wind, opt, mat_result)) {

    printf("\nFailed to write basical_information.\n");
    return false;
  }
  printf("Finished basical information\n");

  // #warning Basical information is not written here, because nbt library is
  // not avaliable.

  // write end_state
  if (!fractal_bin_file_write_end_state(fp, buffer_A, buffer_A_capacity,
                                        buffer_B, buffer_B_capacity,
                                        mat_result)) {
    printf("\nFailed to write end_state.\n");
    return false;
  }

  // write end_time
  if (!fractal_bin_file_write_collide_time(fp, buffer_A, buffer_A_capacity,
                                           buffer_B, buffer_B_capacity,
                                           mat_result)) {
    printf("\nFailed to write collide_time.\n");
    return false;
  }
  // write end_enrgy
  if (!fractal_bin_file_write_end_energy(fp, buffer_A, buffer_A_capacity,
                                         buffer_B, buffer_B_capacity,
                                         mat_result)) {
    printf("\nFailed to write end_energy.\n");
    return false;
  }
  // write iterate_time
  if (!fractal_bin_file_write_iterate_time(fp, buffer_A, buffer_A_capacity,
                                           buffer_B, buffer_B_capacity,
                                           mat_result)) {
    printf("\nFailed to write iterate_time.\n");
    return false;
  }
  // write search_time
  if (!fractal_bin_file_write_fail_search_time(fp, buffer_A, buffer_A_capacity,
                                               buffer_B, buffer_B_capacity,
                                               mat_result)) {
    printf("\nFailed to write fail_search_time.\n");
    return false;
  }

  fclose(fp);

  return true;
}

bool xz_compress(uint8_t *const dest, const uint8_t *const src,
                 const uint64_t src_bytes, const uint64_t dest_capacity,
                 uint64_t *const dest_bytes) noexcept {
  lzma_stream xzs = LZMA_STREAM_INIT;

  lzma_ret ret;

  ret = lzma_easy_encoder(&xzs, uint32_t(9) | LZMA_PRESET_EXTREME,
                          LZMA_CHECK_CRC64);

  if (ret != LZMA_OK) {
    printf("\nError : lzma_easy_encoder failed with error code %i\n", ret);
    lzma_end(&xzs);
    return false;
  }

  xzs.avail_in = src_bytes;
  xzs.next_in = src;
  xzs.avail_out = dest_capacity;
  xzs.next_out = dest;

  ret = lzma_code(&xzs, LZMA_RUN);

  if (ret != LZMA_OK) {
    printf("\nError : lzma_code failed with error code %i\n", ret);
    lzma_end(&xzs);
    return false;
  }

  if (xzs.avail_in > 0) {
    printf("\nError : compression failed due to enough buffer.\n");
    lzma_end(&xzs);
    return false;
  }

  ret = lzma_code(&xzs, LZMA_FINISH);

  if (ret != LZMA_STREAM_END) {
    printf("\nError : lzma_code failed with error code %i\n", ret);
    lzma_end(&xzs);
    return false;
  }

  *dest_bytes = xzs.total_out;

  lzma_end(&xzs);
  return true;
}

bool fractal_bin_file_write_basical_information(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const libthreebody::input_t &center_input,
    const fractal_utils::center_wind<double> &wind,
    const libthreebody::compute_options &opt,
    const fractal_utils::fractal_map &mat_result) noexcept {
  using namespace libthreebody;
  using namespace fractal_utils;

  // printf("line = %i\n", __LINE__);

  using namespace nbt;

  // printf("line = %i\n", __LINE__);
  // auto &tags = nbt.data->tags;
  TagCompound tags;
  // printf("line = %i\n", __LINE__);
  //  write matrix size
  tags["rows"] = TagLong(mat_result.rows);
  tags["cols"] = TagLong(mat_result.cols);
  // printf("line = %i\n", __LINE__);
  //  write center_input
  {
    std::vector<double> mass(3);
    memcpy(mass.data(), center_input.mass.data(), 3 * sizeof(double));

    std::vector<double> beg_state(18);
    memcpy(beg_state.data(), center_input.beg_state.position.data(),
           9 * sizeof(double));
    memcpy(beg_state.data() + 9, center_input.beg_state.velocity.data(),
           9 * sizeof(double));
    tags["center_input"] = TagCompound{{"mass", TagList(mass)},
                                       {"initial_state", TagList(beg_state)}};
  }
  // printf("line = %i\n", __LINE__);
  //  write window
  tags["window"] =
      TagCompound{{"center", TagCompound{{"x", TagDouble(wind.center[0])},
                                         {"y", TagDouble(wind.center[1])}}},
                  {"x_span", TagDouble(wind.x_span)},
                  {"y_span", TagDouble(wind.y_span)}};
  // printf("line = %i\n", __LINE__);
  //  write compute_options
  tags["compute_options"] =
      TagCompound{{"time_end", TagDouble(opt.time_end)},
                  {"max_relative_error", TagDouble(opt.max_relative_error)},
                  {"step_guess", TagDouble(opt.step_guess)}};

  // printf("line = %i\n", __LINE__);

  nbt::NBT nbt{"basical_information", tags};
  /*
  std::stringstream ss;
  ss.str().reserve(4096);
  // memset(ss.str().data(), 0xFF, 4096);
  ss.clear();
  nbt.encode(ss);
  // printf("line = %i\n", __LINE__);
  printf("size = %i.\n", (int)ss.str().size());
  // printf("line = %i\n", __LINE__);
  data_block blk;
  blk.tag = fractal_binfile_tag::basical_information;
  blk.data = const_cast<char *>(ss.str().data());
  blk.bytes = ss.str().size();
  */

  void *buffer_B_cur = buffer_B;
  nbt.encode(&buffer_B_cur, (void *)(buffer_B_capacity + buffer_B));

  data_block blk;
  blk.tag = fractal_binfile_tag::basical_information;
  blk.data = buffer_B;
  blk.bytes = (uint8_t *)buffer_B_cur - buffer_B;

  printf("size = %i.\n", (int)blk.bytes);

  // printf("line = %i\n", __LINE__);
  bool success = write_data_block(fp, blk);
  if (!success) {
    printf("\nError : write_data_block failed.\n");
    return false;
  }

  // printf("line = %i\n", __LINE__);
  return true;
}

bool fractal_bin_file_write_end_state(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept {
  // return true;
  using namespace libthreebody;
  using namespace fractal_utils;
  uint8_t *buffer_A_dest = buffer_A;
  for (int i = 0; i < mat_result.element_count(); i++) {
    memcpy(buffer_A_dest, mat_result.at<result_t>(i).end_state.data(),
           18 * sizeof(double));
    buffer_A_dest += 18 * sizeof(double);
  }

  data_block blk;
  blk.tag = fractal_binfile_tag::matrix_end_state;
  blk.data = buffer_B;
  blk.bytes = 0;

  bool success = xz_compress(buffer_B, buffer_A, buffer_A_dest - buffer_A,
                             buffer_B_capacity, &blk.bytes);

  if (!success) {
    printf("\nError : xz compression failed.\n");
    fclose(fp);
    return false;
  }

  success = write_data_block(fp, blk);
  if (!success) {
    printf("\nError : write_data_block failed.\n");
    printf("blk.bytes = %lu\n", blk.bytes);
    fclose(fp);
    return false;
  }

  return true;
}

bool fractal_bin_file_write_collide_time(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept {
  using namespace libthreebody;
  using namespace fractal_utils;
  double *dest_A = reinterpret_cast<double *>(buffer_A);

  for (int i = 0; i < mat_result.element_count(); i++) {
    *dest_A = mat_result.at<result_t>(i).end_time;
    dest_A++;
  }

  const size_t buffer_A_bytes = reinterpret_cast<uint8_t *>(dest_A) - buffer_A;

  data_block blk;
  blk.tag = fractal_binfile_tag::matrix_collide_time;
  blk.data = buffer_B;
  blk.bytes = 0;

  bool success;
  success = xz_compress(buffer_B, buffer_A, buffer_A_bytes, buffer_B_capacity,
                        &blk.bytes);

  if (!success) {
    printf("\nError : xz compression failed.\n");
    fclose(fp);
    return false;
  }

  success = write_data_block(fp, blk);
  if (!success) {
    printf("\nError : write_data_block failed.\n");
    fclose(fp);
    return false;
  }
  return true;
}

bool fractal_bin_file_write_end_energy(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept {
  using namespace libthreebody;
  using namespace fractal_utils;
  double *dest_A = reinterpret_cast<double *>(buffer_A);

  for (int i = 0; i < mat_result.element_count(); i++) {
    *dest_A = mat_result.at<result_t>(i).end_energy;
    dest_A++;
  }

  const size_t buffer_A_bytes = reinterpret_cast<uint8_t *>(dest_A) - buffer_A;

  data_block blk;
  blk.tag = fractal_binfile_tag::matrix_end_energy;
  blk.data = buffer_B;
  blk.bytes = 0;

  bool success;
  success = xz_compress(buffer_B, buffer_A, buffer_A_bytes, buffer_B_capacity,
                        &blk.bytes);

  if (!success) {
    printf("\nError : xz compression failed.\n");
    fclose(fp);
    return false;
  }

  success = write_data_block(fp, blk);
  if (!success) {
    printf("\nError : write_data_block failed.\n");
    fclose(fp);
    return false;
  }
  return true;
}

bool fractal_bin_file_write_iterate_time(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept {
  using namespace libthreebody;
  using namespace fractal_utils;
  int *dest_A = reinterpret_cast<int *>(buffer_A);

  for (int i = 0; i < mat_result.element_count(); i++) {
    *dest_A = mat_result.at<result_t>(i).iterate_times;
    dest_A++;
  }

  const size_t buffer_A_bytes = reinterpret_cast<uint8_t *>(dest_A) - buffer_A;

  data_block blk;
  blk.tag = fractal_binfile_tag::matrix_iterate_time;
  blk.data = buffer_B;
  blk.bytes = 0;

  bool success;
  success = xz_compress(buffer_B, buffer_A, buffer_A_bytes, buffer_B_capacity,
                        &blk.bytes);

  if (!success) {
    printf("\nError : xz compression failed.\n");
    fclose(fp);
    return false;
  }

  success = write_data_block(fp, blk);
  if (!success) {
    printf("\nError : write_data_block failed.\n");
    fclose(fp);
    return false;
  }
  return true;
}

bool fractal_bin_file_write_fail_search_time(
    FILE *const fp, uint8_t *const buffer_A, const size_t buffer_A_capacity,
    uint8_t *const buffer_B, const size_t buffer_B_capacity,
    const fractal_utils::fractal_map &mat_result) noexcept {
  using namespace libthreebody;
  using namespace fractal_utils;
  int *dest_A = reinterpret_cast<int *>(buffer_A);

  for (int i = 0; i < mat_result.element_count(); i++) {
    *dest_A = mat_result.at<result_t>(i).fail_search_times;
    dest_A++;
  }

  const size_t buffer_A_bytes = reinterpret_cast<uint8_t *>(dest_A) - buffer_A;

  data_block blk;
  blk.tag = fractal_binfile_tag::matrix_iterate_fail_time;
  blk.data = buffer_B;
  blk.bytes = 0;

  bool success;
  success = xz_compress(buffer_B, buffer_A, buffer_A_bytes, buffer_B_capacity,
                        &blk.bytes);

  if (!success) {
    printf("\nError : xz compression failed.\n");
    fclose(fp);
    return false;
  }

  success = write_data_block(fp, blk);
  if (!success) {
    printf("\nError : write_data_block failed.\n");
    fclose(fp);
    return false;
  }
  return true;
}