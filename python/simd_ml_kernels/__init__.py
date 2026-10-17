"""simd-ml-kernels: hand-vectorized x86 SIMD kernels for float32 NumPy arrays.

GEMM (FP32, FP16-input, BF16-input), GeLU, SiLU, ReLU, softmax and LayerNorm,
implemented in C++17 with AVX2 / AVX-512 intrinsics. The compiled extension is
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
    get_num_threads,
    is_aligned,
    layer_norm,
    relu,
    set_num_threads,
    sgemm,
    sgemm_bf16,
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
    "get_num_threads",
    "is_aligned",
    "layer_norm",
    "relu",
    "set_num_threads",
    "sgemm",
    "sgemm_bf16",
    "sgemm_f16",
    "silu",
    "softmax",
]
