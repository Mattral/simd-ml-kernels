"""
Type stubs for simd_ml_kernels._core, the compiled extension module.

All kernels operate on float32 NumPy arrays (float16, BF16 or uint8/int8
inputs for the low-precision GEMMs, float32 output) and release the GIL during
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
    *,
    trans_b: bool = False,
) -> NDArray[np.float32]:
    """
    FP16-input GEMM: C[float32] = alpha * A[float16] @ op(B)[float16] + beta * C[float32]

    trans_b=True: B is stored [N, K] (torch.nn.Linear's weight layout) and
    C = A @ B.T; up to 8 rows of A stream the rows of B in place (decode GEMV).

    A and B must have dtype=np.float16 (IEEE 754 half-precision, 16-bit).
    Output C is always float32 — FP16 accumulation loses precision at K > 64.

    Runs on every CPU: F16C conversion in the AVX2/AVX-512 kernels (once per
    element while packing, then the FP32 micro-kernel), exact software
    conversion in the scalar ones. The error bound is the FP32 GEMM's.

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
    """Set the most OpenMP threads one sgemm call uses (default: OMP_NUM_THREADS,
    else one per logical CPU). Small products and calls from inside an OpenMP
    parallel region use fewer; results are bitwise the same for any count.
    Without OpenMP (build_info()['openmp'] False), n > 1 warns (RuntimeWarning)."""
    ...


def get_num_threads() -> int:
    """Return the most OpenMP threads one sgemm call uses (1 without OpenMP)."""
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

    Available on Intel Cooper Lake, Sapphire Rapids and later, and AMD Zen 4
    and later. Not required for sgemm_bf16. Where True and SIMD_ML_BF16_DOT=1
    is set, the AVX-512 table's sgemm_bf16 uses vdpbf16ps for packed
    single-threaded products (off by default: slower than the conversion
    path where measured; BF16 subnormal inputs read as zero).
    """
    ...


def sgemm_bf16(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[NDArray[np.float32]] = None,
    *,
    trans_b: bool = False,
) -> NDArray[np.float32]:
    """
    BF16-input GEMM: C[float32] = alpha * A[bfloat16] @ op(B)[bfloat16] + beta * C[float32]

    trans_b=True: B is stored [N, K] (torch.nn.Linear's weight layout) and
    C = A @ B.T; up to 8 rows of A stream the rows of B in place (decode GEMV).

    A and B store Brain Float 16 values (same exponent range as FP32, 7-bit mantissa).
    Accepts uint16 dtype (explicit bit-cast) or the bfloat16 dtype of ml_dtypes.
    Output C is always float32.

    Runs on every CPU. BF16→FP32 is exact (zero-extend and shift left by 16),
    so the error bound is the FP32 GEMM's for the BF16 values. The AVX2 and
    AVX-512 tables convert once per element while packing and run the FP32
    micro-kernel. Opt-in: with AVX512_BF16 and SIMD_ML_BF16_DOT=1, packed
    single-threaded products use vdpbf16ps (BF16 subnormal inputs read as
    zero, subnormal results flushed).

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


def sgemm_u8s8(
    A: NDArray[np.uint8],
    B: NDArray[np.int8],
    a_scale: float,
    a_zero_point: int,
    b_scales: NDArray[np.float32],
    C: Optional[NDArray[np.float32]] = None,
    *,
    trans_b: bool = False,
) -> NDArray[np.float32]:
    """
    INT8 GEMM with per-channel scales, float32 output:
    C[i, j] = a_scale * b_scales[j] * sum_k (A[i, k] - a_zero_point) * B[k, j]

    A: uint8 [M, K] (activations, per-tensor scale and zero point); B: int8
    [K, N], or [N, K] with trans_b=True (weights, symmetric, one scale per
    output column); b_scales: float32 [N]. As PyTorch's quantized linear, but
    the result is float32, not requantized. C is overwritten.

    Paths (build_info()['u8s8_path']): avx_vnni / avx512_vnni (vpdpbusd, exact
    int32 sums per K block), fp32_convert (AVX2 without VNNI: about the FP32
    GEMM's speed, a quarter of its operand memory), scalar (exact int64 sums).
    One row of A with trans_b (M = 1) reads B in place on every x86 path.
    Error bound: |C - C_exact| <= 2*(K+2)*2^-24 * |a_scale*b_scales[j]| *
    sum_k |A[i,k] - a_zero_point| * |B[k,j]|. Single-threaded.
    """
    ...


def quantize_q8_0(W: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Quantize float32 weights [..., K] (K % 32 == 0) to ggml's Q8_0 blocks,
    byte-identical to block_q8_0: per 32 values a float16 scale
    d = max|w|/127 (little-endian), then 32 int8 q = roundf(w/d), as ggml's
    quantize_row_q8_0_ref. Returns uint8 [..., K // 32 * 34].

    Raises ArgumentError if K % 32 != 0, or W has a NaN, an infinity, or a
    block whose scale overflows float16 (|w| >= 8321040).
    """
    ...


def gemv_q8_0(x: NDArray[np.float32], Wq: NDArray[np.uint8], N: int, K: int) -> NDArray[np.float32]:
    """
    y[n] = sum_k x[k] * W[n, k] for W [N, K] given as quantize_q8_0 blocks
    (Wq: N*K/32*34 uint8), x float32 [K]. x is quantized to Q8 blocks once
    per call (float32 scales); integer block dot products (vpdpbusd /
    vpmaddubsw / scalar; build_info()['q8_0_path']), float32 sums. Error
    bound against the exact sum of the quantized operands:
    2*(K/32 + 4)*2^-24 * sum_b |dw_b*dx_b| * sum_i |qw*qx|. x with NaN or
    an infinity: all NaN. Single-threaded. Returns float32 [N].
    """
    ...


def quantize_q4_0(W: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Quantize float32 weights [..., K] (K % 32 == 0) to ggml's Q4_0 blocks,
    byte-identical to block_q4_0: per 32 values a float16 scale d = max/-8
    (max: the value of largest magnitude, little-endian), then 16 bytes of
    4-bit q = min(15, trunc(w/d + 8.5)), values j and 16 + j in the low and
    high nibble of byte j, as ggml's quantize_row_q4_0_ref; value =
    d*(q - 8). Returns uint8 [..., K // 32 * 18].

    Raises ArgumentError if K % 32 != 0, or W has a NaN, an infinity, or a
    block whose scale overflows float16 (|w| >= 524160).
    """
    ...


def gemv_q4_0(x: NDArray[np.float32], Wq: NDArray[np.uint8], N: int, K: int) -> NDArray[np.float32]:
    """
    y[n] = sum_k x[k] * W[n, k] for W [N, K] given as quantize_q4_0 blocks
    (Wq: N*K/32*18 uint8), x float32 [K]. x is quantized to Q8 blocks once
    per call (float32 scales); integer block dot products (vpdpbusd /
    vpmaddubsw / scalar; the path build_info()['q8_0_path'] names), float32
    sums. Error bound against the exact sum of the quantized operands:
    2*(K/32 + 4)*2^-24 * sum_b |dw_b*dx_b| * sum_i |(qw - 8)*qx|. x with
    NaN or an infinity: all NaN. Single-threaded. Returns float32 [N].
    """
    ...


def is_aligned(arr: np.ndarray) -> bool:
    """Return True if the array's data buffer is 64-byte aligned."""
    ...


# ─── Private: used by `python -m simd_ml_kernels bench` and `info` ────────────

def _measure_core_ghz(seconds: float = 0.05) -> float: ...
def _cpu_caches() -> list[dict[str, int | str]]: ...
def _stream_triad(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    c: NDArray[np.float32],
    s: float,
    threads: int = 1,
) -> None: ...
def _stream_read(a: NDArray[np.float32], threads: int = 1, streams: int = 4) -> float: ...
def _stream_read_rows(a: NDArray[np.float32], rows: int, row_floats: int, prefetch: bool = False) -> float: ...
def _set_isa(isa: str) -> None: ...
def _sgemm_u8s8_convert(
    A: NDArray[np.uint8],
    B: NDArray[np.int8],
    a_scale: float,
    a_zero_point: int,
    b_scales: NDArray[np.float32],
    C: Optional[NDArray[np.float32]] = None,
    *,
    trans_b: bool = False,
) -> NDArray[np.float32]: ...
