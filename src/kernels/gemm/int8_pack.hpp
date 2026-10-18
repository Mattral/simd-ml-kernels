#pragma once

/**
 * int8_pack.hpp — packing for the VNNI INT8 GEMMs (avxvnni_gemm.cpp,
 * avx512vnni_gemm.cpp). vpdpbusd multiplies 4 consecutive uint8 of one
 * operand with 4 int8 of the other and adds the 4 products to an int32 lane,
 * so K is taken in quads:
 *
 *   A micro-panel (MR rows): Ap[q·MR + r] = the 4 bytes A(r, 4q .. 4q+3) as one
 *     32-bit word, broadcast to every lane by the kernel; rows past mc and
 *     k past kc are 0.
 *   B micro-panel (NR columns): Bp[q·NR + c] = the 4 bytes op(B)(4q .. 4q+3, c)
 *     (lane c), then NR int32 column sums Σ_k op(B)(k, c) at Bp[kq·NR + c],
 *     used to subtract A's zero point: Σ (a − zp)·b = Σ a·b − zp·Σ b.
 *     Columns past nc and k past kc are 0.
 *
 * Everything here is in an anonymous namespace on purpose: each unit is
 * compiled with its own ISA flags, and an inline function emitted out of line
 * by two units would be one linker-chosen copy (tests/check_isa_units.py).
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace simd_ml {
namespace gemm {
namespace {

/** Number of k-quads for kc values of k. */
constexpr int kquads(int kc) { return (kc + 3) / 4; }

/** Words in one packed B micro-panel of NR columns (quads, then column sums). */
template <int NR>
constexpr std::size_t b_panel_words(int kc) { return static_cast<std::size_t>(kquads(kc) + 1) * NR; }

/** pack_a_u8<MR> — A[0..mc)[0..kc) (row-major, lda) into MR-row micro-panels of k-quads. */
template <int MR>
void pack_a_u8(const std::uint8_t* A, std::ptrdiff_t lda, int mc, int kc, std::uint32_t* Ap) noexcept {
    const int kq = kquads(kc), full = kc / 4;
    for (int i = 0; i < mc; i += MR) {
        const int rows = mc - i < MR ? mc - i : MR;
        std::uint32_t* dst = Ap + static_cast<std::ptrdiff_t>(i / MR) * MR * kq;
        for (int r = 0; r < MR; ++r) {
            if (r >= rows) {
                for (int q = 0; q < kq; ++q) dst[q * MR + r] = 0;
                continue;
            }
            const std::uint8_t* src = A + (i + r) * lda;
            for (int q = 0; q < full; ++q) std::memcpy(&dst[q * MR + r], src + 4 * q, 4);
            if (full < kq) {   // last, partial quad
                std::uint8_t t[4] = {0, 0, 0, 0};
                std::memcpy(t, src + 4 * full, static_cast<std::size_t>(kc - 4 * full));
                std::memcpy(&dst[full * MR + r], t, 4);
            }
        }
    }
}

/**
 * pack_b_s8_panel<NR> — quads of columns [0, cols) of one B micro-panel, any
 * layout, byte by byte (partial panels, the K tail, and the AVX-512 unit);
 * op(B)(k, c) = B[k·ldb + c], or B[c·ldb + k] with trans. Leaves the column
 * sums to the caller. q0: first quad to fill.
 */
template <int NR>
void pack_b_s8_panel(const std::int8_t* B, std::ptrdiff_t ldb, bool trans, int kc, int cols, int q0,
                     std::uint32_t* dst) noexcept {
    const int kq = kquads(kc);
    for (int q = q0; q < kq; ++q)
        for (int c = 0; c < NR; ++c) {
            std::uint8_t t[4] = {0, 0, 0, 0};
            for (int h = 0; h < 4; ++h) {
                const int k = 4 * q + h;
                if (c < cols && k < kc)
                    t[h] = static_cast<std::uint8_t>(trans ? B[c * ldb + k] : B[k * ldb + c]);
            }
            std::memcpy(&dst[q * NR + c], t, 4);
        }
}

}  // namespace
}  // namespace gemm
}  // namespace simd_ml
