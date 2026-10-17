/**
 * kernels_avx512.cpp — the AVX-512 kernel table. An AVX-512 unit
 * (-mavx512f -mavx512dq -mavx512bw -mavx512vl); dispatch.cpp uses it only on
 * CPUs that also pass the AVX2 checks and have OS ZMM state.
 *
 * Only FP32 GEMM has an AVX-512 path. Every other entry is the AVX2 kernel:
 * an AVX-512 variant goes in only when it is measured to be faster.
 */

#include "../dispatch/kernels.hpp"
#include "activations/activations.hpp"
#include "gemm/avx2_gemm_packed.hpp"
#include "gemm/bf16_gemm.hpp"
#include "gemm/f16_gemm.hpp"
#include "intrinsic_gelu.hpp"

#ifndef SIMD_ML_UNIT_FLAGS
#  define SIMD_ML_UNIT_FLAGS "unknown"
#endif

namespace simd_ml {

const KernelTable& kernel_table_avx512() noexcept {
    static const KernelTable t = {
        Isa::avx512, "avx512", SIMD_ML_UNIT_FLAGS, "8x16 (8 ZMM accumulators)",
        &gemm::sgemm_packed_avx512, &gemm::sgemm_f16_avx2, &gemm::sgemm_bf16_avx2,
        &gelu_erf_forward_avx2, &gelu_forward_avx2,
        &activations::relu_avx2, &activations::silu_avx2,
        &activations::softmax_row_avx2, &activations::layer_norm_avx2,
    };
    return t;
}

}  // namespace simd_ml
