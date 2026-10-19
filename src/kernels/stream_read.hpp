#pragma once

/**
 * stream_read.hpp — read-only bandwidth loops for `bench`'s roof (the
 * private _core._stream_read_rows), in the access pattern of the decode
 * GEMVs: the array viewed as rows of row_floats values, `rows` rows walked
 * together along the row with 32-byte loads (2 YMM per row per step, as the
 * FP32 GEMV, docs/DESIGN.md §2.7), then the next group of rows; with
 * prefetch, the same row of the next group is prefetched one line per 64
 * bytes read, as the Q8_0 GEMV does (§2.8). AVX2 unit (stream_read_avx2.cpp):
 * call only where the AVX2 table is usable.
 */

#include <cstddef>

namespace simd_ml {

/** Sum of p[0..n) read as described above; rows is 4 or 8. Elements past the last full group are summed plainly. */
double stream_read_rows_avx2(const float* p, std::ptrdiff_t n, int rows, std::ptrdiff_t row_floats,
                             bool prefetch) noexcept;

}  // namespace simd_ml
