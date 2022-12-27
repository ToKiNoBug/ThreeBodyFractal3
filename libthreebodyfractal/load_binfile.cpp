#include <lzma.h>
#include <stdio.h>
#include <stdlib.h>

#include "threebodyfractal.h"

#include <map>

#include <nbt.hpp>

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

    *rows_dest = (size_t)std::get<nbt::TagLong>(info.at("rows"));
  }

  if (cols_dest != nullptr) {
    if (!info.contains("cols")) {
      printf("\nError : function fractal_bin_file_get_information failed to "
             "find tag \"cols\" : no such tag.\n");
      return false;
    }

    *rows_dest = (size_t)std::get<nbt::TagLong>(info.at("cols"));
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
// ret = lzma_auto_decoder(lzma_stream *strm, uint64_t memlimit, uint32_t
// flags)
#warning not finished

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
    size_t rows, cols;
    bool ok = fractal_bin_file_get_information(binfile, &rows, &cols);

    if (!ok) {
      printf("\nError : function fractal_bin_file_get_end_state failed to get "
             "information.\n");
      return false;
    }

    if (rows != end_state_dest->rows || cols != end_state_dest->cols) {
      printf(
          "\nError : function fractal_bin_file_get_end_state examine_map_size "
          "failed. Matrix size mismatch.\n");
      return false;
    }
  }
#warning not finished yet.
  return true;
}