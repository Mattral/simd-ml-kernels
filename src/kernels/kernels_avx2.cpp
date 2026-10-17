/**
 * kernels_avx2.cpp — the AVX2 kernel table. An AVX2 unit (-mavx2 -mfma -mf16c);
 * dispatch.cpp uses it only on CPUs with AVX2, FMA, F16C and OS YMM state.
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

const KernelTable& kernel_table_avx2() noexcept {
    static const KernelTable t = {
        Isa::avx2, "avx2", SIMD_ML_UNIT_FLAGS, "8x8 (8 YMM accumulators)",
        &gemm::sgemm_packed_avx2, &gemm::sgemm_f16_avx2, &gemm::sgemm_bf16_avx2,
        &gelu_erf_forward_avx2, &gelu_forward_avx2,
        &activations::relu_avx2, &activations::silu_avx2,
        &activations::softmax_row_avx2, &activations::layer_norm_avx2,
    };
    return t;
}

}  // namespace simd_ml
