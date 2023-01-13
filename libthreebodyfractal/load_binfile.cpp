#include <lzma.h>
#include <stdio.h>
#include <stdlib.h>

#include "libthreebodyfractal.h"

const fractal_utils::data_block *
find_data_block_noduplicate(const fractal_utils::binfile &binfile,
                            libthreebody::fractal_binfile_tag tag) noexcept {
  const fractal_utils::data_block *ret = nullptr;

  for (const auto &blk : binfile.blocks) {
    if (blk.tag == tag) {
      if (ret != nullptr) {
        printf("\nError : multiple datablocks have tag %i, which is not "
               "allowed.\n",
               int(tag));
        return nullptr;
      }
      ret = &blk;
    }
  }

  if (ret == nullptr) {
    printf("\nError : failed to find any datablock with tag %i.\n", int(tag));
    return nullptr;
  }

  return ret;
}

#include <libNBT++/io/stream_reader.h>
#include <libNBT++/nbt_tags.h>
#include <libNBT++/tag.h>
#include <sstream>

bool libthreebody::fractal_bin_file_get_information(
    const fractal_utils::binfile &binfile, size_t *const rows_dest,
    size_t *const cols_dest, input_t *const center_input_dest,
    fractal_utils::center_wind<double> *const wind_dest,
    compute_options *const opt_dest) noexcept {
  const fractal_utils::data_block *info_block = nullptr;
  for (const auto &blk : binfile.blocks) {
    if (blk.tag == fractal_binfile_tag::basical_information) {
      if (info_block != nullptr) {
        printf(
            "\nError : multiple datablocks have tag %i (basical_information)\n",
            int(fractal_binfile_tag::basical_information));
        return false;
      }

      info_block = &blk;
    }
  }

  if (info_block == nullptr) {
    printf("\nError : failed to find any datablock with tag %i "
           "(basical_information).\n",
           int(fractal_binfile_tag::basical_information));
    return false;
  }

  std::stringstream ss;

  ss.write((const char *)info_block->data, info_block->bytes);

  nbt::io::stream_reader sr(ss);

  auto pair = sr.read_tag();

  if (pair.first != "basical_information") {
  }

  const nbt::tag_compound &info = pair.second->as<nbt::tag_compound>();

  if (rows_dest != nullptr) {
    if (!info.has_key("rows") ||
        info.at("rows").get_type() != nbt::tag_type::Long) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"rows\" : no such tag.\n");
      return false;
    }

    *rows_dest = info.at("rows").as<nbt::tag_long>();
  }

  if (cols_dest != nullptr) {
    if (!info.has_key("cols") ||
        info.at("cols").get_type() != nbt::tag_type::Long) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"cols\" : no such tag.\n");
      return false;
    }

    *cols_dest = info.at("cols").as<nbt::tag_long>();
  }

  if (center_input_dest != nullptr) {
    if (!info.has_key("center_input") ||
        info.at("center_input").get_type() != nbt::tag_type::Compound) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"center_input\" : no such tag.\n");
      return false;
    }

    const nbt::tag_compound &ci =
        info.at("center_input").as<nbt::tag_compound>();

    if (!ci.has_key("initial_state") ||
        ci.at("initial_state").get_type() != nbt::tag_type::List) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"center_input/initial_state\" : no such tag.\n");
      return false;
    }
    const nbt::tag_list &is = ci.at("initial_state").as<nbt::tag_list>();
    if (is.size() != 18 || is.el_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed "
             "because size of initial_state is not 18 but %i, or element type "
             "is not double\n",
             int(is.size()));
      return false;
    }
    std::array<double, 18> isd;
    for (int i = 0; i < 18; i++) {
      isd[i] = is[i].as<nbt::tag_double>();
    }

    memcpy(center_input_dest->beg_state.position.data(), isd.data(),
           sizeof(double) * 9);
    memcpy(center_input_dest->beg_state.velocity.data(), isd.data() + 9,
           sizeof(double) * 9);

    if (!ci.has_key("mass") ||
        ci.at("mass").get_type() != nbt::tag_type::List) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"center_input/mass\" : no such tag.\n");
      return false;
    }
    const nbt::tag_list &m = ci.at("mass").as<nbt::tag_list>();

    if (m.size() != 3 || m.el_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed "
             "because size of mass is not 3 but %i, or element type "
             "is not double\n",
             int(m.size()));
      return false;
    }
    std::array<double, 3> md;

    for (int i = 0; i < 3; i++) {
      md[i] = m[i].as<nbt::tag_double>();
    }

    memcpy(center_input_dest->mass.data(), md.data(), sizeof(double) * 3);
  }

  if (wind_dest != nullptr) {
    if (!info.has_key("window") ||
        info.at("window").get_type() != nbt::tag_type::Compound) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window\" : no such tag.\n");
      return false;
    }

    const nbt::tag_compound &w = info.at("window").as<nbt::tag_compound>();

    if (!w.has_key("x_span") ||
        w.at("x_span").get_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/x_span\" : no such tag.\n");
      return false;
    }
    wind_dest->x_span = w.at("x_span").as<nbt::tag_double>();

    if (!w.has_key("y_span") ||
        w.at("y_span").get_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/y_span\" : no such tag.\n");
      return false;
    }
    wind_dest->y_span = w.at("y_span").as<nbt::tag_double>();

    if (!w.has_key("center") ||
        w.at("center").get_type() != nbt::tag_type::Compound) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/center\" : no such tag.\n");
      return false;
    }
    const nbt::tag_compound &wc = w.at("center").as<nbt::tag_compound>();
    if (!wc.has_key("x") || wc.at("x").get_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/center/x\" : no such tag.\n");
      return false;
    }
    wind_dest->center[0] = wc.at("x").as<nbt::tag_double>();
    if (!wc.has_key("y") || wc.at("y").get_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/center/y\" : no such tag.\n");
      return false;
    }
    wind_dest->center[1] = wc.at("y").as<nbt::tag_double>();
  }

  if (opt_dest != nullptr) {
    if (!info.has_key("compute_option") ||
        info.at("compute_option").get_type() != nbt::tag_type::Compound) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option\" : no such tag.\n");
      return false;
    }
    const nbt::tag_compound &co =
        info.at("compute_option").as<nbt::tag_compound>();

    if (!co.has_key("max_relative_error") ||
        co.at("max_relative_error").get_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option/max_relative_error\" : no such tag.\n");
      return false;
    }
    opt_dest->max_relative_error =
        co.at("max_relative_error").as<nbt::tag_double>();

    if (!co.has_key("step_guess") ||
        co.at("step_guess").get_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option/step_guess\" : no such tag.\n");
      return false;
    }
    opt_dest->step_guess = co.at("step_guess").as<nbt::tag_double>();

    if (!co.has_key("time_end") ||
        co.at("time_end").get_type() != nbt::tag_type::Double) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option/time_end\" : no such tag.\n");
      return false;
    }
    opt_dest->time_end = co.at("time_end").as<nbt::tag_double>();
  }

  return true;
}

bool xz_decompress(const uint8_t *const src, const uint64_t src_bytes,
                   uint8_t *const dest, const uint64_t dest_capacity,
                   uint64_t *const dest_bytes) {
  lzma_stream xzs = LZMA_STREAM_INIT;

  lzma_ret ret;
  ret = lzma_stream_decoder(&xzs, UINT64_MAX, LZMA_CONCATENATED);

  if (ret != LZMA_OK) {
    printf("\nError : function xz_decompress failed to initialize decoder "
           "stream with error code %i.\n",
           ret);
    lzma_end(&xzs);
    return false;
  }

  xzs.next_in = src;
  xzs.avail_in = src_bytes;
  xzs.next_out = dest;
  xzs.avail_out = dest_capacity;

  ret = lzma_code(&xzs, LZMA_RUN);
  if (ret != LZMA_OK) {
    printf("\nError : function xz_decompress failed to decode with error code "
           "%i.\n",
           ret);
    lzma_end(&xzs);
    return false;
  }

  ret = lzma_code(&xzs, LZMA_FINISH);

  if (ret != LZMA_STREAM_END && ret != LZMA_OK) {
    printf("\nError : function xz_decompress failed to finish with error code "
           "%i.\n",
           ret);
    lzma_end(&xzs);
    return false;
  }

  *dest_bytes = xzs.total_out;

  lzma_end(&xzs);

  return true;
}

bool check_information(const fractal_utils::binfile &binfile,
                       const fractal_utils::fractal_map &dest_matrix) noexcept {
  size_t rows = 0, cols = 0;
  bool ok =
      libthreebody::fractal_bin_file_get_information(binfile, &rows, &cols);

  if (!ok) {
    printf("\nError : function check_information failed to get "
           "information.\n");
    return false;
  }

  if (rows != dest_matrix.rows || cols != dest_matrix.cols) {
    printf("\nError : function check_information failed. Matrix size mismatch. "
           "Result from binfile is (%llu,%llu), but size of matrix is "
           "(%llu,%llu).\n",
           rows, cols, dest_matrix.rows, dest_matrix.cols);
    return false;
  }
  return true;
}

bool libthreebody::fractal_bin_file_get_end_state(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_state_dest,
    const bool examine_map_size) noexcept {
  if (end_state_dest->element_bytes != sizeof(state_t)) {
    printf(
        "\nError : end_state_dest->element_bytes is %i, but sizeof(state_t) is "
        "%i.\n",
        int(end_state_dest->element_bytes), int(sizeof(state_t)));
    return false;
  }

  if (examine_map_size) {
    bool ok = check_information(binfile, *end_state_dest);
    if (!ok) {
      return false;
    }
  }

  const fractal_utils::data_block *const state_blk =
      find_data_block_noduplicate(
          binfile, libthreebody::fractal_binfile_tag::matrix_end_state);
  if (state_blk == nullptr) {
    return false;
  }
  size_t decompressed_bytes = 0;
  bool ok = true;
  ok = xz_decompress((uint8_t *)state_blk->data, state_blk->bytes,
                     (uint8_t *)end_state_dest->data,
                     end_state_dest->byte_count(), &decompressed_bytes);
  if (!ok) {
    return false;
  }

  if (decompressed_bytes !=
      end_state_dest->element_count() * sizeof(double) * 18) {
    printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
           "it is %zu.\n",
           end_state_dest->element_count() * sizeof(double) * 18,
           decompressed_bytes);
    return false;
  }

  const double *const data = (double *)end_state_dest->data;

  for (int64_t idx = end_state_dest->element_count() - 1; idx >= 0; idx--) {
    memmove(end_state_dest->at<state_t>(idx).position.data(), data + idx * 18,
            sizeof(double[9]));
    memmove(end_state_dest->at<state_t>(idx).velocity.data(),
            data + idx * 18 + 9, sizeof(double[9]));
  }

  return true;
}

bool libthreebody::fractal_bin_file_get_end_energy(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_state_energy,
    const bool examine_map_size) noexcept {
  if (end_state_energy->element_bytes != sizeof(double)) {
    printf(
        "\nError : end_state_energy->element_bytes is %u, but sizeof(double) "
        "is %zu.\n",
        end_state_energy->element_bytes, sizeof(double));
    return false;
  }

  if (examine_map_size) {
    bool ok = check_information(binfile, *end_state_energy);
    if (!ok) {
      return false;
    }
  }

  const fractal_utils::data_block *const state_blk =
      find_data_block_noduplicate(
          binfile, libthreebody::fractal_binfile_tag::matrix_end_energy);
  if (state_blk == nullptr) {
    return false;
  }

  size_t decompressed_bytes = 0;
  bool ok = true;
  ok = xz_decompress((uint8_t *)state_blk->data, state_blk->bytes,
                     (uint8_t *)end_state_energy->data,
                     end_state_energy->byte_count(), &decompressed_bytes);
  if (!ok) {
    return false;
  }
  if (decompressed_bytes != end_state_energy->byte_count()) {
    printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
           "it is %zu.\n",
           end_state_energy->byte_count(), decompressed_bytes);
    return false;
  }

  return true;
}

bool libthreebody::fractal_bin_file_get_collide_time(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_time_dest,
    const bool examine_map_size) noexcept {
  if (end_time_dest->element_bytes != sizeof(double)) {
    printf("\nError : end_time_dest->element_bytes is %u, but sizeof(double) "
           "is %zu.\n",
           end_time_dest->element_bytes, sizeof(double));
    return false;
  }

  if (examine_map_size) {
    bool ok = check_information(binfile, *end_time_dest);
    if (!ok) {
      return false;
    }
  }

  const fractal_utils::data_block *const state_blk =
      find_data_block_noduplicate(
          binfile, libthreebody::fractal_binfile_tag::matrix_collide_time);
  if (state_blk == nullptr) {
    return false;
  }

  size_t decompressed_bytes = 0;
  bool ok = true;
  ok = xz_decompress((uint8_t *)state_blk->data, state_blk->bytes,
                     (uint8_t *)end_time_dest->data,
                     end_time_dest->byte_count(), &decompressed_bytes);
  if (!ok) {
    return false;
  }
  if (decompressed_bytes != end_time_dest->byte_count()) {
    printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
           "it is %zu.\n",
           end_time_dest->byte_count(), decompressed_bytes);
    return false;
  }

  return true;
}

bool libthreebody::fractal_bin_file_get_iterate_time(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_iterate_time_dest,
    const bool examine_map_size) noexcept {
  if (end_iterate_time_dest->element_bytes != sizeof(int)) {
    printf("\nError : end_iterate_time_dest->element_bytes is %u, but "
           "sizeof(int) "
           "is %zu.\n",
           end_iterate_time_dest->element_bytes, sizeof(int));
    return false;
  }

  if (examine_map_size) {
    bool ok = check_information(binfile, *end_iterate_time_dest);
    if (!ok) {
      return false;
    }
  }

  const fractal_utils::data_block *const state_blk =
      find_data_block_noduplicate(
          binfile, libthreebody::fractal_binfile_tag::matrix_iterate_time);
  if (state_blk == nullptr) {
    return false;
  }

  size_t decompressed_bytes = 0;
  bool ok = true;
  ok = xz_decompress((uint8_t *)state_blk->data, state_blk->bytes,
                     (uint8_t *)end_iterate_time_dest->data,
                     end_iterate_time_dest->byte_count(), &decompressed_bytes);
  if (!ok) {
    return false;
  }
  if (decompressed_bytes != end_iterate_time_dest->byte_count()) {
    printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
           "it is %zu.\n",
           end_iterate_time_dest->byte_count(), decompressed_bytes);
    return false;
  }

  return true;
}

bool libthreebody::fractal_bin_file_get_iterate_fail_time(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const end_iterate_fail_time_dest,
    const bool examine_map_size) noexcept {
  if (end_iterate_fail_time_dest->element_bytes != sizeof(int)) {
    printf("\nError : end_iterate_fail_time_dest->element_bytes is %u, but "
           "sizeof(int) "
           "is %zu.\n",
           end_iterate_fail_time_dest->element_bytes, sizeof(int));
    return false;
  }

  if (examine_map_size) {
    bool ok = check_information(binfile, *end_iterate_fail_time_dest);
    if (!ok) {
      return false;
    }
  }

  const fractal_utils::data_block *const state_blk =
      find_data_block_noduplicate(
          binfile, libthreebody::fractal_binfile_tag::matrix_iterate_time);
  if (state_blk == nullptr) {
    return false;
  }

  size_t decompressed_bytes = 0;
  bool ok = true;
  ok = xz_decompress((uint8_t *)state_blk->data, state_blk->bytes,
                     (uint8_t *)end_iterate_fail_time_dest->data,
                     end_iterate_fail_time_dest->byte_count(),
                     &decompressed_bytes);
  if (!ok) {
    return false;
  }
  if (decompressed_bytes != end_iterate_fail_time_dest->byte_count()) {
    printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
           "it is %zu.\n",
           end_iterate_fail_time_dest->byte_count(), decompressed_bytes);
    return false;
  }

  return true;
}

bool libthreebody::fractal_bin_file_get_result(
    const fractal_utils::binfile &binfile,
    fractal_utils::fractal_map *const result_dest, void *buffer,
    size_t buffer_capacity, const bool examine_map_size) noexcept {
  if (result_dest->element_bytes != sizeof(result_t)) {
    printf("\nError : result_dest->element_bytes is %u, but "
           "sizeof(result_t) "
           "is %zu.\n",
           result_dest->element_bytes, sizeof(result_t));
    return false;
  }

  if (examine_map_size) {
    bool ok = check_information(binfile, *result_dest);
    if (!ok) {
      return false;
    }
  }

  if (buffer_capacity < result_dest->element_count() * sizeof(double[18])) {
    printf("\nError : buffer capacity not enough. Expected at least %zu\n",
           result_dest->element_count() * sizeof(double[18]));
    return false;
  }
  {
    const fractal_utils::data_block *blk = nullptr;
    bool ok = true;
    size_t decompressed_bytes = 0;

    //-------------------- end state --------------
    blk = find_data_block_noduplicate(
        binfile, libthreebody::fractal_binfile_tag::matrix_end_state);
    if (blk == nullptr) {
      return false;
    }
    ok = xz_decompress((const uint8_t *)blk->data, blk->bytes,
                       (uint8_t *)buffer, buffer_capacity, &decompressed_bytes);
    if (!ok) {
      return false;
    }
    if (decompressed_bytes !=
        result_dest->element_count() * sizeof(double[18])) {
      printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
             "it is %zu.\n",
             result_dest->element_count() * sizeof(double[18]),
             decompressed_bytes);
      return false;
    }
    for (int idx = 0; idx < result_dest->element_count(); idx++) {
      const double *src = ((double *)buffer) + idx * 18;

      auto &dest = result_dest->at<result_t>(idx).end_state;

      memcpy(dest.position.data(), src, sizeof(double[9]));
      memcpy(dest.velocity.data(), src + 9, sizeof(double[9]));
    }

    //-------------------- end energy --------------
    blk = find_data_block_noduplicate(binfile,
                                      fractal_binfile_tag::matrix_end_energy);
    if (blk == nullptr) {
      return false;
    }
    ok = xz_decompress((const uint8_t *)blk->data, blk->bytes,
                       (uint8_t *)buffer, buffer_capacity, &decompressed_bytes);
    if (!ok) {
      return false;
    }
    if (decompressed_bytes != result_dest->element_count() * sizeof(double)) {
      printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
             "it is %zu.\n",
             result_dest->element_count() * sizeof(double), decompressed_bytes);
      return false;
    }
    for (int idx = 0; idx < result_dest->element_count(); idx++) {
      const double *src = ((double *)buffer);
      result_dest->at<result_t>(idx).end_energy = src[idx];
    }

    //-------------------- end time --------------
    blk = find_data_block_noduplicate(binfile,
                                      fractal_binfile_tag::matrix_collide_time);
    if (blk == nullptr) {
      return false;
    }
    ok = xz_decompress((const uint8_t *)blk->data, blk->bytes,
                       (uint8_t *)buffer, buffer_capacity, &decompressed_bytes);
    if (!ok) {
      return false;
    }
    if (decompressed_bytes != result_dest->element_count() * sizeof(double)) {
      printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
             "it is %zu.\n",
             result_dest->element_count() * sizeof(double), decompressed_bytes);
      return false;
    }
    for (int idx = 0; idx < result_dest->element_count(); idx++) {
      const double *src = ((double *)buffer);
      result_dest->at<result_t>(idx).end_time = src[idx];
    }

    //-------------------- iterate time --------------
    blk = find_data_block_noduplicate(binfile,
                                      fractal_binfile_tag::matrix_iterate_time);
    if (blk == nullptr) {
      return false;
    }
    ok = xz_decompress((const uint8_t *)blk->data, blk->bytes,
                       (uint8_t *)buffer, buffer_capacity, &decompressed_bytes);
    if (!ok) {
      return false;
    }
    if (decompressed_bytes != result_dest->element_count() * sizeof(int)) {
      printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
             "it is %zu.\n",
             result_dest->element_count() * sizeof(int), decompressed_bytes);
      return false;
    }
    for (int idx = 0; idx < result_dest->element_count(); idx++) {
      const int *src = ((int *)buffer);
      result_dest->at<result_t>(idx).iterate_times = src[idx];
    }

    //-------------------- iterate fail time --------------
    blk = find_data_block_noduplicate(
        binfile, fractal_binfile_tag::matrix_iterate_fail_time);
    if (blk == nullptr) {
      return false;
    }
    ok = xz_decompress((const uint8_t *)blk->data, blk->bytes,
                       (uint8_t *)buffer, buffer_capacity, &decompressed_bytes);
    if (!ok) {
      return false;
    }
    if (decompressed_bytes != result_dest->element_count() * sizeof(int)) {
      printf("\nError : decompressed bytes mismatch. Should be %zu but infact "
             "it is %zu.\n",
             result_dest->element_count() * sizeof(int), decompressed_bytes);
      return false;
    }
    for (int idx = 0; idx < result_dest->element_count(); idx++) {
      const int *src = ((int *)buffer);
      result_dest->at<result_t>(idx).fail_search_times = src[idx];
    }
  }

  return true;
}