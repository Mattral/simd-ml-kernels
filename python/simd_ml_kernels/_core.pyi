"""
Type stubs for simd_ml_kernels._core, the compiled extension module.

All kernels operate on float32 NumPy arrays and release the GIL during
computation. Arrays must be C-contiguous; call np.ascontiguousarray() if needed.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


# ─── Linear Algebra ───────────────────────────────────────────────────────────

def sgemm(
    A: NDArray[np.float32],
    B: NDArray[np.float32],
    C: Optional[NDArray[np.float32]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    isa: str = "",
) -> NDArray[np.float32]:
    """
    SIMD GEMM: C = alpha * A @ B + beta * C  (float32, in-place or allocates).

    Parameters
    ----------
    A     : ndarray[float32], shape [M, K]
    B     : ndarray[float32], shape [K, N]
    C     : ndarray[float32], shape [M, N], optional (allocated if None)
    alpha : float scalar, default 1.0
    beta  : float scalar, default 0.0 (0 = overwrite C)
    isa   : str, default "" — forces this call to use a specific kernel:
            "" (auto-detect, default), "avx2", or "avx512". Requesting
            "avx512" on hardware without AVX-512F+DQ raises RuntimeError
            (check avx512_available() first). "scalar" is a recognized
            value but raises RuntimeError ("not yet implemented") rather
            than silently falling back. This override applies only to
            this call, not subsequent ones.

    Returns
    -------
    ndarray[float32], shape [M, N]
    """
    ...


def sgemm_f16(
    A: NDArray[np.float16],
    B: NDArray[np.float16],
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    FP16-input GEMM: C[float32] = alpha * A[float16] @ B[float16] + beta * C[float32]

    A and B must have dtype=np.float16 (IEEE 754 half-precision, 16-bit).
    Output C is always float32 — FP16 accumulation loses precision at K > 64.

    Requires F16C instructions (Haswell+, AMD Piledriver+). Check
    f16c_available() before calling; raises RuntimeError if absent.

    Parameters
    ----------
    A     : ndarray[float16], shape [M, K], C-contiguous
    B     : ndarray[float16], shape [K, N], C-contiguous
    alpha : float, default 1.0
    beta  : float, default 0.0  (0 = overwrite C, 1 = accumulate into C)
    C     : ndarray[float32], shape [M, N], optional (allocated if None)

    Returns
    -------
    ndarray[float32], shape [M, N]

    Example
    -------
    >>> if simd_ml_kernels.f16c_available():
    ...     A = np.random.randn(64, 256).astype(np.float16)
    ...     B = np.random.randn(256, 64).astype(np.float16)
    ...     C = simd_ml_kernels.sgemm_f16(A, B)   # float32 result
    """
    ...


class GEMMConfig:
    """
    Callable GEMM configuration object.

    Stores default alpha, beta, and isa values for repeated GEMM calls.
    Tile parameters (tile_m, tile_n, tile_k, mr, nr) are accepted for API
    completeness and future auto-tuning integration; they are currently
    informational only.

    Examples
    --------
    >>> gemm = GEMMConfig(alpha=2.0, beta=0.5, isa="avx2")
    >>> C = gemm(A, B)
    """

    alpha: float
    beta: float
    isa: str
    tile_m: int
    tile_n: int
    tile_k: int
    mr: int
    nr: int

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        isa: str = "",
        tile_m: int = 128,
        tile_n: int = 2048,
        tile_k: int = 256,
        mr: int = 8,
        nr: int = 8,
    ) -> None: ...

    def __call__(
        self,
        A: NDArray[np.float32],
        B: NDArray[np.float32],
        C: Optional[NDArray[np.float32]] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> NDArray[np.float32]: ...

    def __repr__(self) -> str: ...


# ─── Activation Functions ─────────────────────────────────────────────────────

def gelu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Out-of-place AVX2 GeLU (tanh approximation, max error < 5e-6).
    Returns a new array; input x is unchanged.
    """
    ...


def gelu_inplace(x: NDArray[np.float32]) -> None:
    """
    In-place AVX2 GeLU: x[:] = GeLU(x).
    Modifies x directly; returns None.
    """
    ...


def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return ReLU(x) = max(0, x) computed with AVX2 vectorization."""
    ...


def silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return SiLU(x) = x * sigmoid(x) using AVX2 vectorization."""
    ...


def softmax(
    x: NDArray[np.float32],
    axis: int = -1,
) -> NDArray[np.float32]:
    """Return numerically stable softmax over the specified axis (default: last)."""
    ...


def layer_norm(
    x: NDArray[np.float32],
    gamma: Optional[NDArray[np.float32]] = None,
    beta: Optional[NDArray[np.float32]] = None,
    eps: float = 1e-5,
) -> NDArray[np.float32]:
    """
    Layer Normalization over the last axis.

    output = (x - mean) / sqrt(var + eps) * gamma + beta

    Parameters
    ----------
    x     : ndarray[float32], any shape
    gamma : ndarray[float32], shape [last_dim], optional scale
    beta  : ndarray[float32], shape [last_dim], optional shift
    eps   : float, numerical stability (default 1e-5)
    """
    ...


# ─── Threading ────────────────────────────────────────────────────────────────

def set_num_threads(n: int) -> None:
    """Set the number of OpenMP threads used by GEMM (requires SIMD_ML_OPENMP build)."""
    ...


def get_num_threads() -> int:
    """Return the current OpenMP thread count used by GEMM."""
    ...


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def build_info() -> dict[str, str | bool]:
    """Return a dict with compiled ISA, FMA support, alignment, and build timestamp."""
    ...


def detected_isa() -> str:
    """Return the runtime-detected ISA label: 'avx512', 'avx2', 'sse42', or 'scalar'."""
    ...


def avx512_available() -> bool:
    """
    Return True if this CPU supports AVX-512F+DQ (required by sgemm_packed's
    8x16 micro-kernel), independent of any isa= override on a given call.

    Check this before requesting isa='avx512' explicitly to avoid a
    RuntimeError on unsupported hardware.
    """
    ...


def f16c_available() -> bool:
    """
    Return True if this CPU supports F16C instructions (_mm256_cvtph_ps).

    F16C is available on Intel Haswell (2013) and later, and on AMD Piledriver
    (2012) and later — essentially all CPUs that also have AVX2. Check this
    before calling sgemm_f16() to get a clean error instead of SIGILL.
    """
    ...


def bf16_avx512bf16_available() -> bool:
    """
    Return True if this CPU supports AVX-512 BF16 (vdpbf16ps instruction).

    Available on Intel Ice Lake-SP (Xeon, 2021+), Sapphire Rapids,
    and AMD Zen4+ (EPYC Genoa). NOT required for sgemm_bf16 (which uses the
    AVX2 zero-extend path). Indicates availability of the higher-throughput
    vdpbf16ps packing path planned for v1.0.
    """
    ...


def sgemm_bf16(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    BF16-input GEMM: C[float32] = alpha * A[bfloat16] @ B[bfloat16] + beta * C[float32]

    A and B store Brain Float 16 values (same exponent range as FP32, 7-bit mantissa).
    Accepts uint16 dtype (explicit bit-cast) or bfloat16 dtype (numpy>=2.0 / ml_dtypes).
    Output C is always float32.

    Requires only AVX2 (Haswell+) — no special AVX-512 BF16 flag needed.
    BF16→FP32 uses the zero-extend+shift trick (3 integer instructions per 8 values).

    Parameters
    ----------
    A     : array [M, K], dtype=uint16 or bfloat16, C-contiguous
    B     : array [K, N], dtype=uint16 or bfloat16, C-contiguous
    alpha : float, default 1.0
    beta  : float, default 0.0  (0 = overwrite C, 1 = accumulate)
    C     : ndarray[float32], shape [M, N], optional

    Returns
    -------
    ndarray[float32], shape [M, N]
    """
    ...


def is_aligned(arr: np.ndarray) -> bool:
    """Return True if the array's data buffer is 64-byte aligned."""
    ...
