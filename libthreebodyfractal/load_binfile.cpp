#include <lzma.h>
#include <stdio.h>
#include <stdlib.h>

#include "threebodyfractal.h"

bool libthreebody::fractal_bin_file_get_information(
    const fractal_utils::binfile &binfile, size_t *const rows_dest,
    size_t *const cols_dest, input_t *const center_input_dest,
    fractal_utils::center_wind<double> *const wind_dest,
    compute_options *const opt_dest) noexcept {
#warning not finished yet. waitting for nbt parsing lib
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
      printf(
          "\nError : function fractal_bin_file_get_end_state failed to get "
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