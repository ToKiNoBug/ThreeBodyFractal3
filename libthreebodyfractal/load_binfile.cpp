#include <lzma.h>
#include <stdio.h>
#include <stdlib.h>

#include "threebodyfractal.h"

#include <map>

#include <nbt.hpp>

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

  nbt::NBT nbt(
      info_block->data,
      (const void *)((const uint8_t *)info_block->data + info_block->bytes));

  const nbt::TagCompound &info = nbt.data->tags;

  if (rows_dest != nullptr) {
    if (!info.contains("rows")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"rows\" : no such tag.\n");
      return false;
    }

    *rows_dest = std::get<nbt::TagLong>(info.at("rows"));
  }

  if (cols_dest != nullptr) {
    if (!info.contains("cols")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"cols\" : no such tag.\n");
      return false;
    }

    *rows_dest = std::get<nbt::TagLong>(info.at("cols"));
  }

  if (center_input_dest != nullptr) {
    if (!info.contains("center_input")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"center_input\" : no such tag.\n");
      return false;
    }

    const nbt::TagCompound &ci =
        std::get<nbt::TagCompound>(info.at("center_input"));

    if (!ci.contains("initial_state")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"center_input/initial_state\" : no such tag.\n");
      return false;
    }
    const nbt::TagList &is = std::get<nbt::TagList>(ci.at("initial_state"));
    const std::vector<nbt::TagDouble> &isd =
        std::get<std::vector<nbt::TagDouble>>(is);

    if (isd.size() != 18) {
      printf("\nError : function fractal_bin_file_get_information failed "
             "because size of initial_state is not 18 but %i\n",
             int(isd.size()));
      return false;
    }

    memcpy(center_input_dest->beg_state.position.data(), isd.data(),
           sizeof(double) * 9);
    memcpy(center_input_dest->beg_state.velocity.data(), isd.data() + 9,
           sizeof(double) * 9);

    if (!ci.contains("mass")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"center_input/mass\" : no such tag.\n");
      return false;
    }
    const nbt::TagList &m = std::get<nbt::TagList>(ci.at("mass"));
    const std::vector<nbt::TagDouble> &md =
        std::get<std::vector<nbt::TagDouble>>(m);

    if (md.size() != 3) {
      printf("\nError : function fractal_bin_file_get_information failed "
             "because size of mass is not 3 but %i\n",
             int(md.size()));
      return false;
    }

    memcpy(center_input_dest->mass.data(), md.data(), sizeof(double) * 3);
  }

  if (wind_dest != nullptr) {
    if (!info.contains("window")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window\" : no such tag.\n");
      return false;
    }

    const nbt::TagCompound &w = std::get<nbt::TagCompound>(info.at("window"));

    if (!w.contains("x_span")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/x_span\" : no such tag.\n");
      return false;
    }
    wind_dest->x_span = std::get<nbt::TagDouble>(w.at("x_span"));

    if (!w.contains("y_span")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/y_span\" : no such tag.\n");
      return false;
    }
    wind_dest->y_span = std::get<nbt::TagDouble>(w.at("y_span"));

    if (!w.contains("center")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/center\" : no such tag.\n");
      return false;
    }
    const nbt::TagCompound &wc = std::get<nbt::TagCompound>(w.at("center"));
    if (!wc.contains("x")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/center/x\" : no such tag.\n");
      return false;
    }
    wind_dest->center[0] = std::get<nbt::TagDouble>(wc.at("x"));
    if (!wc.contains("y")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"window/center/y\" : no such tag.\n");
      return false;
    }
    wind_dest->center[1] = std::get<nbt::TagDouble>(wc.at("y"));
  }

  if (opt_dest != nullptr) {
    if (!info.contains("compute_option")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option\" : no such tag.\n");
      return false;
    }
    const nbt::TagCompound &co =
        std::get<nbt::TagCompound>(info.at("compute_option"));

    if (!co.contains("max_relative_error")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option/max_relative_error\" : no such tag.\n");
      return false;
    }
    opt_dest->max_relative_error =
        std::get<nbt::TagDouble>(co.at("max_relative_error"));

    if (!co.contains("step_guess")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option/step_guess\" : no such tag.\n");
      return false;
    }
    opt_dest->step_guess = std::get<nbt::TagDouble>(co.at("step_guess"));

    if (!co.contains("time_end")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"compute_option/time_end\" : no such tag.\n");
      return false;
    }
    opt_dest->time_end = std::get<nbt::TagDouble>(co.at("time_end"));
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
    printf("\nError : decompressed bytes mismatch. Should be %llu but infact "
           "it is %llu.\n",
           end_state_dest->element_count() * sizeof(double) * 18,
           decompressed_bytes);
    return false;
  }

  const double *const data = (double *)end_state_dest->data;

  for (int64_t idx = end_state_dest->element_count() - 1; idx >= 0; idx--) {
    memmove(end_state_dest->at<state_t>(idx).position.data(),
            data + idx * sizeof(double[18]), sizeof(double[9]));
    memmove(end_state_dest->at<state_t>(idx).velocity.data(),
            data + idx * sizeof(double[18]) + sizeof(double[9]),
            sizeof(double[9]));
  }

  return true;
}