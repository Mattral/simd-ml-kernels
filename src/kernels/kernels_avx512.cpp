/**
 * kernels_avx512.cpp — the AVX-512 kernel table. An AVX-512 unit
 * (-mavx512f -mavx512dq -mavx512bw -mavx512vl); dispatch.cpp uses it only on
 * CPUs that also pass the AVX2 checks and have OS ZMM state.
 *
 * Only the GEMMs (and the Q8_0 GEMV's AVX512_VNNI dot) have AVX-512 paths: FP32, and FP16/BF16 through the same
 * 12×32 kernel with the conversion in packing (BF16 with vdpbf16ps where the
 * CPU has AVX512_BF16). Every other entry is the AVX2 kernel: an AVX-512
 * variant goes in only when it is measured to be faster.
 */

#include "../dispatch/kernels.hpp"
#include "activations/activations.hpp"
#include "gemm/avx2_gemm_packed.hpp"
#include "gemm/bf16_gemm.hpp"
#include "gemm/f16_gemm.hpp"
#include "gemm/int8_gemm.hpp"
#include "intrinsic_gelu.hpp"
#include "quant/q8_0.hpp"

#ifndef SIMD_ML_UNIT_FLAGS
#  define SIMD_ML_UNIT_FLAGS "unknown"
#endif

namespace simd_ml {

namespace quant {

// Q8_0 GEMV: AVX512_VNNI (q8_0_avx512vnni.cpp) where the CPU has it, else the
// AVX2 table's choice (AVX-VNNI or vpmaddubsw).
void gemv_q8_0_avx512(int N, int K, const float* x, const std::uint8_t* Wq, float* y) {
#ifdef SIMD_ML_HAVE_AVX512VNNI
    if (gemm::avx512vnni_available()) {
        gemv_q8_0_run(N, K, x, Wq, y, &q8_0_rows_avx512vnni);
        return;
    }
#endif
    gemv_q8_0_avx2(N, K, x, Wq, y);
}

}  // namespace quant

const KernelTable& kernel_table_avx512() noexcept {
    static const KernelTable t = {
        Isa::avx512, "avx512", SIMD_ML_UNIT_FLAGS, "12x32 (24 ZMM accumulators)",
        &gemm::sgemm_blocking_avx512,
        &gemm::sgemm_packed_avx512, &gemm::sgemm_f16_avx512, &gemm::sgemm_bf16_avx512, &gemm::sgemm_u8s8_avx512,
        &quant::gemv_q8_0_avx512,
        &gelu_erf_forward_avx2, &gelu_forward_avx2,
        &activations::relu_avx2, &activations::silu_avx2,
        &activations::softmax_row_avx2, &activations::layer_norm_avx2,
    };
    return t;
}

}  // namespace simd_ml
