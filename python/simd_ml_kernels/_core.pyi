"""
Type stubs for simd_ml_kernels._core, the compiled extension module.

All kernels operate on float32 NumPy arrays and release the GIL during
computation. Arrays must be C-contiguous; call np.ascontiguousarray() if needed.
"""

from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray


# ─── Errors ───────────────────────────────────────────────────────────────────

class DTypeError(TypeError, RuntimeError):
    """An array argument has the wrong dtype. The RuntimeError base (the type
    raised before 0.3) is kept until 0.5.0."""

class ArgumentError(ValueError, RuntimeError):
    """An argument has the wrong shape, layout (C-contiguity, writeability) or
    value. The RuntimeError base (the type raised before 0.3) is kept until 0.5.0."""


# ─── Linear Algebra ───────────────────────────────────────────────────────────

def sgemm(
    A: NDArray[np.float32],
    B: NDArray[np.float32],
    C: Optional[NDArray[np.float32]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    isa: str = "",
    *,
    trans_a: bool = False,
    trans_b: bool = False,
) -> NDArray[np.float32]:
    """
    SIMD GEMM: C = alpha * op(A) @ op(B) + beta * C  (float32, in-place or
    allocates); op(X) is X, or X.T with trans_a / trans_b (BLAS transa/transb).

    Parameters
    ----------
    A     : ndarray[float32], shape [M, K] (or [K, M] with trans_a)
    B     : ndarray[float32], shape [K, N] (or [N, K] with trans_b)
    C     : ndarray[float32], shape [M, N], optional (allocated if None)
    alpha : float scalar, default 1.0
    beta  : float scalar, default 0.0 (0 = overwrite C)
    isa   : str, default "" — the kernels chosen for this CPU. "scalar",
            "avx2" or "avx512" run that kernel table for this call only;
            one this build or CPU cannot run raises RuntimeError
            (build_info()["isa_usable"] lists the ones that work).
    trans_a, trans_b : bool, keyword-only, default False — use A.T / B.T as
            stored, without a copy; x @ W.T with W [out, in] is
            sgemm(x, W, trans_b=True).

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

    Runs on every CPU: F16C conversion in the AVX2 kernels, exact software
    conversion in the scalar ones.

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
    >>> A = np.random.randn(64, 256).astype(np.float16)
    >>> B = np.random.randn(256, 64).astype(np.float16)
    >>> C = simd_ml_kernels.sgemm_f16(A, B)   # float32 result
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

def gelu(x: NDArray[np.float32], approximate: str = "none") -> NDArray[np.float32]:
    """
    GeLU, same definitions as torch.nn.functional.gelu. approximate="none"
    (default): exact 0.5*x*(1 + erf(x/sqrt(2))), max abs error <= 2e-6;
    approximate="tanh": tanh formula, <= 2e-6 vs that formula (4.7e-4 from
    exact). Returns a new array; x is unchanged.
    """
    ...


def gelu_inplace(x: NDArray[np.float32], approximate: str = "none") -> None:
    """
    In-place GeLU: x[:] = gelu(x, approximate). x must be writeable.
    """
    ...


def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return ReLU(x) = max(0, x)."""
    ...


def silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).

    Error vs float64: |err| <= 1e-35 + 1e-6*|SiLU(x)| for every finite x
    (tested; measured 2.9e-7 relative on the AVX2 kernel). silu(+inf) = +inf,
    silu(-inf) = NaN as in PyTorch, and x < -88 gives -0 (|SiLU(x)| < 1.5e-36).
    """
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

def build_info() -> dict[str, Any]:
    """Build details: version, git_commit, compiler, isa_compiled and isa_usable
    (space-separated table names), isa_flags (table -> compiler flags),
    runtime_isa (the active table), kernel_flags, sgemm_tile and sgemm_blocking
    ({kc, mc, nc}, derived from the CPU caches; 0 = not blocked) of that table,
    openmp, alignment, build_timestamp."""
    ...


def detected_isa() -> str:
    """Return the kernel table every call uses: 'avx512', 'avx2' or 'scalar'."""
    ...


def avx512_available() -> bool:
    """
    Return True if this build includes the AVX-512 kernels and this CPU and
    OS can run them (AVX-512F/DQ/BW/VL, OS ZMM state).
    """
    ...


def f16c_available() -> bool:
    """
    Return True if this CPU supports F16C (vcvtph2ps), which the AVX2 kernels
    require. sgemm_f16() runs without it (scalar kernels).
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

    Runs on every CPU. BF16→FP32 is exact: zero-extend and shift left by 16.

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


# ─── Private: used by `python -m simd_ml_kernels bench` and `info` ────────────

def _measure_core_ghz(seconds: float = 0.05) -> float: ...
def _cpu_caches() -> list[dict[str, int | str]]: ...
def _set_isa(isa: str) -> None: ...
