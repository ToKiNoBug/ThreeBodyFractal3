#ifndef THREEBODYFRACTAL3_THREEBODYFRACTAL_H
#define THREEBODYFRACTAL3_THREEBODYFRACTAL_H

#include "libthreebody.h"

namespace libthreebody {

void compute_many(const input_t *const src, result_t *const dest,
                  const uint64_t count, const compute_options &opt);
}

#endif // THREEBODYFRACTAL3_THREEBODYFRACTAL_H