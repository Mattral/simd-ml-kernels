# Python API Reference

**Module**: `simd_ml_kernels`  
**Import**: `import simd_ml_kernels`  
**Type stubs**: `python/simd_ml_kernels/_core.pyi` (IDE autocompletion)

All kernels operate on **float32 NumPy arrays**, release the **GIL** during computation (threading-safe), and require **C-contiguous** memory layout. Call `numpy.ascontiguousarray(arr)` if needed.

---

## Linear Algebra

### `sgemm(A, B, C=None, alpha=1.0, beta=0.0) → np.ndarray`

Compute `C = alpha * A @ B + beta * C` (single-precision GEMM).

| Parameter | Type | Description |
|---|---|---|
| `A` | `ndarray[float32]`, shape `[M, K]` | Left operand |
| `B` | `ndarray[float32]`, shape `[K, N]` | Right operand |
| `C` | `ndarray[float32]`, shape `[M, N]`, optional | Output (allocated if `None`) |
| `alpha` | `float` | Scale factor for `A @ B` (default `1.0`) |
| `beta` | `float` | Scale factor for existing `C` (`0.0` = overwrite) |

Returns the `[M, N]` result array (same object as `C` if provided).

```python
import numpy as np, simd_ml_kernels

A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)

# Simple GEMM (allocates output)
C = simd_ml_kernels.sgemm(A, B)

# In-place with scaling: C = 2*A@B + 0.5*C
simd_ml_kernels.sgemm(A, B, C, alpha=2.0, beta=0.5)

# Transposed operands, no copy (BLAS transa/transb): a linear layer with W
# stored [out_features, in_features], as torch.nn.Linear stores it
W = np.random.randn(128, 512).astype(np.float32)
y = simd_ml_kernels.sgemm(A, W, trans_b=True)          # A @ W.T, shape [256, 128]
```

`trans_a` / `trans_b` (keyword-only) mean A is stored `[K, M]` / B is stored
`[N, K]`. The transpose is read while packing the panels (8×8 register
transposes for B), or, for `trans_b` with `M ≤ 16`, by a dot-product path
along rows of A and B. `examples/linear_trans_b.py` runs both and checks them
against PyTorch.

**Implementation**: a 6×16 AVX2 register tile (12 accumulators) run on A and B in place for small products and inside the Goto/BLIS 5 loops with packed panels otherwise, block sizes from the CPU caches (`build_info()["sgemm_blocking"]`); 12×32 (24 accumulators) on AVX-512; a plain loop in the scalar table. See `docs/DESIGN.md` §2.

### `sgemm_u8s8(A, B, a_scale, a_zero_point, b_scales, C=None, *, trans_b=False) → np.ndarray`

INT8 GEMM with per-channel scales, float32 output:
`C[i, j] = a_scale · b_scales[j] · Σ_k (A[i, k] − a_zero_point) · B[k, j]`.

| Parameter | Type | Description |
|---|---|---|
| `A` | `ndarray[uint8]`, shape `[M, K]` | Quantized activations (one scale and zero point for the tensor) |
| `B` | `ndarray[int8]`, shape `[K, N]`, or `[N, K]` with `trans_b=True` | Quantized weights, symmetric (zero point 0) |
| `a_scale` | `float` | Scale of `A` (finite) |
| `a_zero_point` | `int` in `[0, 255]` | Zero point of `A` |
| `b_scales` | `ndarray[float32]`, shape `[N]` | One scale per output column |
| `C` | `ndarray[float32]`, shape `[M, N]`, optional | Output, overwritten (allocated if `None`) |
| `trans_b` | `bool`, keyword-only | `B` stored `[N, K]`, as `torch.nn.Linear` stores its weight |

This is the arithmetic of PyTorch's quantized linear (quint8 input, per-channel
qint8 weight), except that the result stays float32 instead of being
requantized. Error bound on every path:
`|C − C_exact| ≤ 2·(K+2)·2⁻²⁴·|a_scale·b_scales[j]|·Σ_k |A[i,k] − a_zero_point|·|B[k,j]|`.

```python
x = np.random.randn(8, 256).astype(np.float32)
W = np.random.randn(64, 256).astype(np.float32)               # [out, in]
xs, zp = (x.max() - x.min()) / 255, 128
xq = np.clip(np.round(x / xs) + zp, 0, 255).astype(np.uint8)
ws = (np.abs(W).max(axis=1) / 127).astype(np.float32)         # per output channel
Wq = np.round(W / ws[:, None]).astype(np.int8)
y = simd_ml_kernels.sgemm_u8s8(xq, Wq, xs, zp, ws, trans_b=True)   # ≈ x @ W.T
```

**Implementation** (`build_info()["u8s8_path"]`): `avx_vnni` / `avx512_vnni`:
`vpdpbusd` on 6×16 / 12×32 int32 tiles, exact int32 sums within each K
block (up to 4096 values), the zero point subtracted through column sums of
B taken while packing; `fp32_convert` (AVX2 CPUs without VNNI): the inputs
converted to float32 while packing and the FP32 kernel, about the FP32
GEMM's speed with a quarter of its operand memory; `scalar`: exact int64
sums. Single-threaded. See `docs/DESIGN.md` §4.2.

### `GEMMConfig` — callable GEMM configuration object

```python
class GEMMConfig:
    def __init__(self, alpha=1.0, beta=0.0, isa="",
                 tile_m=128, tile_n=2048, tile_k=256, mr=8, nr=8): ...
    def __call__(self, A, B, C=None, alpha=None, beta=None) -> np.ndarray: ...
```

Stores default `alpha`, `beta`, and `isa` for repeated GEMM calls. Tile
parameters (`tile_m`, `tile_n`, `tile_k`, `mr`, `nr`) are accepted for API
completeness and are validated (must be positive), but do not currently
change dispatch — the compiled kernel uses fixed compile-time constants.
Auto-tuned tile selection is planned (`docs/ROADMAP.md §v0.9`).

```python
gemm = simd_ml_kernels.GEMMConfig(alpha=2.0, beta=0.5, isa="avx2")
C = gemm(A, B)                      # uses stored alpha=2.0, beta=0.5, forces AVX2
C = gemm(A, B, alpha=1.0, beta=0.0) # per-call alpha/beta override (isa stays "avx2")
print(gemm)                          # GEMMConfig(alpha=2.0, beta=0.5, isa='avx2')
```

`isa` must be one of `""`, `"scalar"`, `"avx2"`, `"avx512"`: an invalid
value raises `ValueError` at construction time (`GEMMConfig`) or
`ArgumentError` (a `ValueError`) from `sgemm()`. `sgemm()` accepts the same
argument:

```python
C = simd_ml_kernels.sgemm(A, B)                 # "" = the table chosen for this CPU
C = simd_ml_kernels.sgemm(A, B, isa="scalar")   # portable C++ kernel
C = simd_ml_kernels.sgemm(A, B, isa="avx2")     # 6×16 AVX2 kernel
C = simd_ml_kernels.sgemm(A, B, isa="avx512")   # 12×32 AVX-512 kernel
```

`isa=` runs that kernel table for this call only, e.g. to compare AVX2 and
AVX-512 on one machine. A table that this build does not include, or that
this CPU cannot run, raises `RuntimeError`; `build_info()["isa_usable"]` lists
the usable ones. To change the table for every function and the whole
process, set the environment variable `SIMD_ML_ISA` (`scalar`, `avx2` or
`avx512`; it caps the choice) before importing.

`SIMD_ML_BF16_DOT=1` (also before the first call) lets `sgemm_bf16` use the
`vdpbf16ps` instruction on CPUs with AVX512_BF16. It is off by default: it
was slower than the default conversion path where measured, and it reads BF16
subnormals as zero, so results then differ from other CPUs for tiny values.

---

## Activation Functions

### `gelu(x, approximate="none") → np.ndarray`

Out-of-place GeLU. Input `x` is unchanged.

### `gelu_inplace(x, approximate="none") → None`

In-place GeLU: `x[:] = GeLU(x)`. No allocation. `x` must be writeable.

```python
y = simd_ml_kernels.gelu(x)                       # exact, like torch.nn.functional.gelu(x)
y = simd_ml_kernels.gelu(x, approximate="tanh")   # like gelu(x, approximate='tanh')
simd_ml_kernels.gelu_inplace(x)                   # modifies x in place
```

| `approximate` | formula | max abs error vs float64 (asserted bound / measured) |
|---|---|---|
| `"none"` (default) | `0.5·x·(1 + erf(x/√2))` | ≤ 2e-6 / 9.5e-7 on [−10, 10]; ≤ 1e-6 / 4.0e-7 on [−3, 3] |
| `"tanh"` | `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` | ≤ 2e-6 / 1.5e-6 on [−10, 10] vs the tanh formula |

The two formulas differ from each other by up to 4.7e-4: that is the formula
gap, not implementation error. Measured on a 400,001-point sweep
(`tests/test_precision.py::TestGeLU`, `tests/test_gelu.cpp`). erf is a rational
approximation `x·P(x²)/Q(x²)` with the coefficients of XLA's `ErfImpl32`. For
x < −4, |GeLU(x)| < 1.3e-4 and only the absolute bound is meaningful. Both
variants return −0 for x < −10 (|GeLU(−10)| = 7.6e-23), +inf for +inf, −0 for
−inf and NaN for NaN; the last `n % 8` elements use the same vector kernel
(masked), so results do not depend on array position.

Before 0.3 `gelu` always used the tanh formula; the default is now exact, as in
PyTorch. Pass `approximate="tanh"` for the previous behaviour.

### `relu(x) → np.ndarray`

ReLU activation: `output[i] = max(0, x[i])`. Exact (no approximation).

```python
y = simd_ml_kernels.relu(x)
```

### `silu(x) → np.ndarray`

SiLU/Swish activation: `output[i] = x[i] * sigmoid(x[i])`.  
Implemented as `x / (1 + exp(−x))` with the Cody–Waite `exp` and a correctly rounded division.  
Error vs float64: `|err| ≤ 1e-35 + 1e-6·|SiLU(x)|` for every finite `x` (asserted by the tests and `selftest`; measured 2.9e-7 relative and 1.3e-6 absolute over all finite floats, AVX2 kernel). `silu(+inf) = +inf`, `silu(-inf) = NaN` (as PyTorch), and `x < −88` returns −0 (`|SiLU(x)| < 1.5e-36` there).

```python
y = simd_ml_kernels.silu(x)
```

### `softmax(x, axis=-1) → np.ndarray`

Numerically stable softmax over the specified axis (default: last axis).  
Uses max-subtraction and double-precision accumulation to prevent overflow.

```python
logits = np.random.randn(32, 1000).astype(np.float32)
probs  = simd_ml_kernels.softmax(logits)       # shape [32, 1000], rows sum to 1
```

Currently only the **last axis** (`axis=-1`) is supported. For other axes, transpose first.

Non-finite inputs follow `torch.softmax`: `-inf` entries in an otherwise finite
row (attention masks) get probability 0; a row whose maximum is not finite —
every element `-inf`, any `NaN`, or any `+inf` — returns all `NaN`.

### `layer_norm(x, gamma=None, beta=None, eps=1e-5) → np.ndarray`

Layer normalization over the last axis of `x`:

```
output = (x - mean) / sqrt(var + eps) * gamma + beta
```

`gamma` and `beta` are optional learnable affine parameters with shape matching the last dimension of `x`.

Error vs float64: `|err| ≤ 1e-5 + 1e-4·|ref|` (asserted by the tests and `selftest`) for rows with `|mean| ≤ 30·std`, any row length. Rounding the row mean to float32 costs about `6e-8·|mean|/std` in units of the normalized output, so rows whose mean is many standard deviations from zero exceed that bound; PyTorch's LayerNorm does too (measured 166–1684× the bound at `|mean|/std` = 10⁴–10⁵; this kernel 47–1061×).

```python
x     = np.random.randn(16, 768).astype(np.float32)
gamma = np.ones(768,  dtype=np.float32)   # scale
beta  = np.zeros(768, dtype=np.float32)   # shift

y = simd_ml_kernels.layer_norm(x, gamma=gamma, beta=beta)
# y has zero mean and unit std per row (before affine)
```

**Implementation**: 3 passes (mean → variance → normalize) with the sums in double precision to prevent catastrophic cancellation.

---

## Threading

### `set_num_threads(n: int) → None`

Set the most OpenMP threads one `sgemm` call uses. Until it is called, the
count is OpenMP's default: `OMP_NUM_THREADS`, else one per logical CPU. Small
products (each thread needs a few MFLOP of work), the unpacked small-matrix
paths, and calls made inside an OpenMP parallel region use fewer threads or
one. The result is bitwise the same for every thread count. Other OpenMP code
in the process keeps its own thread count. Built without `-DSIMD_ML_OPENMP`
(the default; `build_info()["openmp"]` is `False`), GEMM is single-threaded
and `n > 1` emits a `RuntimeWarning`.

### `get_num_threads() → int`

Return the most threads one `sgemm` call uses (always 1 without OpenMP).

---

## Diagnostics

### `build_info() → dict`

Build metadata (`python -m simd_ml_kernels info` prints it with the CPU's
features):

| Key | Meaning |
|---|---|
| `version`, `git_commit`, `compiler`, `build_timestamp` | the build; `git_commit` is taken at CMake configure time, `-dirty` marks local changes |
| `isa_compiled` | kernel tables in this build, e.g. `"scalar avx2 avx512"` (only `"scalar"` off x86-64) |
| `isa_usable` | those this CPU and OS can run |
| `isa_flags` | compiler flags of each usable table's units |
| `runtime_isa` | the table every call uses (same as `detected_isa()`) |
| `kernel_flags`, `sgemm_tile` | that table's flags and GEMM micro-kernel shape |
| `openmp` | whether GEMM can use more than one thread |
| `alignment` | allocation alignment of internal buffers |

### `detected_isa() → str`

The kernel table every call uses: `'avx512'`, `'avx2'` or `'scalar'`; chosen
once, on first use (docs/DESIGN.md §6). `sgemm(..., isa=...)` does not change
it.

### `avx512_available() → bool`

`True` if this build includes the AVX-512 kernels and this CPU and OS can run
them (AVX-512F/DQ/BW/VL and OS ZMM state).

### `is_aligned(arr: np.ndarray) → bool`

Return `True` if `arr.data` is 64-byte aligned. Useful for verifying that NumPy arrays created with `np.empty` or `np.zeros` meet the alignment contract.

```python
x = np.empty(1024, dtype=np.float32)
simd_ml_kernels.is_aligned(x)   # True on most NumPy allocations ≥ 64 elements
```

---

## Error Handling

| Problem | Exception | Builtin base |
|---|---|---|
| wrong dtype (including a `C` output that is not float32, and float16 passed to `sgemm_bf16`) | `simd_ml_kernels.DTypeError` | `TypeError` |
| wrong shape/ndim, not C-contiguous, read-only output, bad `axis`, invalid `isa` string | `simd_ml_kernels.ArgumentError` | `ValueError` |
| `isa=` names a table this build or CPU cannot run | `RuntimeError` | — |

Both custom classes also subclass `RuntimeError`, which every argument error
was before 0.3, so `except RuntimeError` keeps working until 0.5.0.

Arrays are never converted silently. In particular an output `C` passed to
`sgemm`/`sgemm_f16`/`sgemm_bf16` must already be a float32, C-contiguous,
writeable `[M, N]` array, because the result is written into it in place.

```python
simd_ml_kernels.gelu(np.array([1.0, 2.0], dtype=np.float64))
# DTypeError: x must be float32 (dtype=np.float32). Call arr.astype(np.float32)...

simd_ml_kernels.relu(arr[::2])
# ArgumentError: x must be C-contiguous. Call numpy.ascontiguousarray(arr)...

simd_ml_kernels.sgemm(np.zeros((4, 8), dtype=np.float32),
                      np.zeros((3, 4), dtype=np.float32))
# ArgumentError: Shape mismatch: A is [4×8] but B is [3×4]
```

---

## Type Stubs

The file `python/simd_ml_kernels/_core.pyi` provides full type annotations for IDE support (Pylance, Pyright, mypy). After `pip install -e .`, type checkers will resolve `simd_ml_kernels` automatically.
