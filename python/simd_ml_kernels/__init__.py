"""simd-ml-kernels: hand-vectorized x86 SIMD kernels for NumPy arrays.

GEMM (FP32, FP16-input, BF16-input, INT8 uint8 x int8), the decode GEMV
(``x @ W.T`` through ``trans_b=True``), GeLU, SiLU, ReLU, softmax and LayerNorm,
implemented in C++17 with AVX2 / AVX-512 / VNNI intrinsics and a portable
scalar version of each, chosen at run time. The compiled extension is
``simd_ml_kernels._core``; this module re-exports its public API.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from ._core import (
    ArgumentError,
    DTypeError,
    GEMMConfig,
    avx512_available,
    bf16_avx512bf16_available,
    build_info,
    detected_isa,
    f16c_available,
    gelu,
    gelu_inplace,
    gemv_q8_0,
    get_num_threads,
    is_aligned,
    layer_norm,
    quantize_q8_0,
    relu,
    set_num_threads,
    sgemm,
    sgemm_bf16,
    sgemm_u8s8,
    sgemm_f16,
    silu,
    softmax,
)

try:
    __version__ = _version("simd-ml-kernels")
except PackageNotFoundError:  # imported from a source tree without installing
    __version__ = "unknown"

__all__ = [
    "ArgumentError",
    "DTypeError",
    "GEMMConfig",
    "avx512_available",
    "bf16_avx512bf16_available",
    "build_info",
    "detected_isa",
    "f16c_available",
    "gelu",
    "gelu_inplace",
    "gemv_q8_0",
    "get_num_threads",
    "is_aligned",
    "layer_norm",
    "quantize_q8_0",
    "relu",
    "set_num_threads",
    "sgemm",
    "sgemm_bf16",
    "sgemm_u8s8",
    "sgemm_f16",
    "silu",
    "softmax",
]
