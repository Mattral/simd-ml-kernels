# Python API Reference

**Module**: `simd_ml_kernels`  
**Import**: `import simd_ml_kernels`  
**Type stubs**: `python/simd_ml_kernels/_core.pyi` (IDE autocompletion)

All kernels operate on **float32 NumPy arrays** (float16, BF16 or uint8/int8 inputs for the low-precision GEMMs, float32 output), release the **GIL** during computation (threading-safe), and require **C-contiguous** memory layout. Call `numpy.ascontiguousarray(arr)` if needed.

---

## Linear Algebra

### `sgemm(A, B, C=None, alpha=1.0, beta=0.0, isa="", *, trans_a=False, trans_b=False) → np.ndarray`

Compute `C = alpha * op(A) @ op(B) + beta * C` (single-precision GEMM).

| Parameter | Type | Description |
|---|---|---|
| `A` | `ndarray[float32]`, shape `[M, K]` (`[K, M]` with `trans_a`) | Left operand |
| `B` | `ndarray[float32]`, shape `[K, N]` (`[N, K]` with `trans_b`) | Right operand |
| `C` | `ndarray[float32]`, shape `[M, N]`, optional | Output (allocated if `None`) |
| `alpha` | `float` | Scale factor for `A @ B` (default `1.0`) |
| `beta` | `float` | Scale factor for existing `C` (`0.0` = overwrite) |
| `isa` | `str` | Kernel table for this call: `""` (default, the one chosen for this CPU), `"scalar"`, `"avx2"`, `"avx512"` (see `GEMMConfig` below) |
| `trans_a`, `trans_b` | `bool`, keyword-only | Use `A.T` / `B.T` as stored, without a copy |

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
transposes for B), or, for `trans_b` with `M ≤ 8`, by the decode GEMV: dot
products along rows of A and B, B read in place (`docs/DESIGN.md` §2.7).
`examples/linear_trans_b.py` runs both and checks them against PyTorch.

### `sgemm_f16(A, B, alpha=1.0, beta=0.0, C=None, *, trans_b=False)` / `sgemm_bf16(...)` → `np.ndarray[float32]`

FP16 (`np.float16`) or BF16 (uint16 bit patterns or `ml_dtypes.bfloat16`)
inputs, FP32 accumulation and output: `C = alpha * A @ op(B) + beta * C`.
Both conversions are exact, so the error bound is the FP32 GEMM's,
`|C − A·op(B)| ≤ 2·K·u·(|A|·|op(B)|)`, u = 2⁻²⁴. With `trans_b=True`
(keyword-only) B is stored `[N, K]`, a weight matrix in `torch.nn.Linear`'s
layout, and op(B) = Bᵀ; up to 8 rows of A take the decode GEMV, which reads
the rows of B in place and converts them in registers (2 bytes per weight).

```python
W = np.random.randn(4096, 4096).astype(np.float16)    # [out, in]
x = np.random.randn(1, 4096).astype(np.float16)
y = simd_ml_kernels.sgemm_f16(x, W, trans_b=True)     # x @ W.T, float32 [1, 4096]
```

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
sums. One row of A with `trans_b=True` (M = 1) reads B in place on every x86
path, with exact integer sums. Single-threaded. See `docs/DESIGN.md` §4.2.

### `quantize_q8_0(W) → np.ndarray[uint8]`

Quantizes float32 weights `[..., K]` (K a multiple of 32) to ggml's Q8_0
format, byte-identical to `block_q8_0`: per 32 values a float16 scale
`d = max|w| / 127` (little-endian) and 32 int8 quants `q = roundf(w / d)`, as
ggml's `quantize_row_q8_0_ref`. Returns `[..., K // 32 * 34]` bytes. The
dequantized value `d16 · q` is within `d · (0.5 + 127·2⁻¹¹) + 2⁻²⁵` of `w`
for blocks with a normal float16 scale (max|w| ≥ 127·2⁻¹⁴).

Raises `ArgumentError` if K % 32 ≠ 0, or W holds a NaN, an infinity, or a
block whose scale overflows float16 (|w| ≥ 8321040).

### `gemv_q8_0(x, Wq, N, K) → np.ndarray[float32]`

`y[n] = Σ_k x[k] · W[n, k]` for W `[N, K]` given as `quantize_q8_0` blocks.

| Parameter | Type | Description |
|---|---|---|
| `x` | `ndarray[float32]`, shape `[K]` | Activations (one token) |
| `Wq` | `ndarray[uint8]`, `N·K/32·34` bytes, any shape | Blocks of W, as `quantize_q8_0(W)` returns them |
| `N`, `K` | `int` | Shape of W; K a multiple of 32 |

x is quantized once per call to the same blocks (float32 scales); each
block's dot product is an exact integer sum, scaled and summed in float32.
Error bound against that sum taken exactly:
`2·(K/32 + 4)·2⁻²⁴·Σ_b |dw_b·dx_b|·Σ_i |qw·qx|`; against the FP32 product,
the 8-bit quantization of W and x dominates (docs/DESIGN.md §2.8). x with a
NaN or an infinity gives NaN everywhere.

```python
W = np.random.randn(4096, 4096).astype(np.float32)   # [out, in]
Wq = simd_ml_kernels.quantize_q8_0(W)               # uint8 [4096, 4352], ggml block_q8_0
x = np.random.randn(4096).astype(np.float32)
y = simd_ml_kernels.gemv_q8_0(x, Wq, 4096, 4096)    # ≈ W @ x
```

**Implementation** (`build_info()["q8_0_path"]`): 8 rows of W read in place
together, the next 8 rows prefetched; `avx_vnni` / `avx512_vnni`:
`vpdpbusd` with ggml's sign trick (|x| · sign(w, x)); `avx2`: `vpmaddubsw` +
`vpmaddwd`; `scalar`: exact int sums. Single-threaded. See
`docs/DESIGN.md` §2.8.

### `quantize_q4_0(W) → np.ndarray[uint8]`

Quantizes float32 weights `[..., K]` (K a multiple of 32) to ggml's Q4_0
format, byte-identical to `block_q4_0`: per 32 values a float16 scale
`d = max / −8` (max: the first value of largest magnitude, sign kept;
little-endian), then 16 bytes of 4-bit quants `q = min(15, trunc(w / d + 8.5))`,
byte j holding value j in its low nibble and value 16 + j in its high
nibble; value = `d · (q − 8)`. As ggml's `quantize_row_q4_0_ref`. Returns
`[..., K // 32 * 18]` bytes. The dequantized value is within
`d · (0.5 + 8·2⁻¹¹) + 2⁻²⁵` of `w` (blocks with a normal float16 scale), except
values with `w / d ≥ 7.5`, which clamp to `q = 15`: within `d · (1 + 8·2⁻¹¹)`.

Raises `ArgumentError` if K % 32 ≠ 0, or W holds a NaN, an infinity, or a
block whose scale overflows float16 (|w| ≥ 524160).

### `gemv_q4_0(x, Wq, N, K) → np.ndarray[float32]`

`y[n] = Σ_k x[k] · W[n, k]` for W `[N, K]` given as `quantize_q4_0` blocks.

| Parameter | Type | Description |
|---|---|---|
| `x` | `ndarray[float32]`, shape `[K]` | Activations (one token) |
| `Wq` | `ndarray[uint8]`, `N·K/32·18` bytes, any shape | Blocks of W, as `quantize_q4_0(W)` returns them |
| `N`, `K` | `int` | Shape of W; K a multiple of 32 |

x is quantized once per call to Q8 blocks, as in `gemv_q8_0` (float32
scales); each block's dot product `Σ (q − 8) · qx` is an exact integer sum,
scaled and summed in float32. Error bound against that sum taken exactly:
`2·(K/32 + 4)·2⁻²⁴·Σ_b |dw_b·dx_b|·Σ_i |(qw − 8)·qx|`; against the FP32
product, the 4-bit weights dominate (docs/DESIGN.md §2.9). x with a NaN or an
infinity gives NaN everywhere.

```python
W = np.random.randn(4096, 4096).astype(np.float32)   # [out, in]
Wq = simd_ml_kernels.quantize_q4_0(W)               # uint8 [4096, 2304], ggml block_q4_0
x = np.random.randn(4096).astype(np.float32)
y = simd_ml_kernels.gemv_q4_0(x, Wq, 4096, 4096)    # ≈ W @ x
```

**Implementation** (the path `build_info()["q8_0_path"]` names): two rows of
W per YMM register (lanes 0–3 and 4–7), 8 rows read in place together, the
next 8 rows prefetched; nibbles unpacked with `vpand` / `vpsrlw`; q is the
unsigned operand of `vpdpbusd` (`vpmaddubsw` on AVX2) and −8·Σx its starting
value, so no sign trick; `scalar`: exact int sums. Single-threaded. See
`docs/DESIGN.md` §2.9.

### `GEMMConfig` — callable GEMM configuration object

```python
class GEMMConfig:
    def __init__(self, alpha=1.0, beta=0.0, isa="",
                 tile_m=128, tile_n=2048, tile_k=256, mr=8, nr=8): ...
    def __call__(self, A, B, C=None, alpha=None, beta=None) -> np.ndarray: ...
```

Stores default `alpha`, `beta`, and `isa` for repeated GEMM calls. Tile
parameters (`tile_m`, `tile_n`, `tile_k`, `mr`, `nr`) are accepted for API
completeness and are validated (must be positive), but they do not change
anything: the micro-kernel shape is fixed per table and the block sizes come
from the CPU's caches (`build_info()["sgemm_tile"]`,
`build_info()["sgemm_blocking"]`; `docs/DESIGN.md` §2.3).

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

**Implementation**: 3 passes (mean → variance → normalize); the statistics are summed in float lanes, shifted by the row's first element, and added into a double every 1024 elements (`docs/DESIGN.md` §3.5).

### `rms_norm(x, weight=None, eps=1e-5) → np.ndarray`

RMS normalization over the last axis of `x`, as Llama's `LlamaRMSNorm`
computes it in float32 (the same order of operations):

```
output = weight * (x * r),   r = 1 / sqrt(mean(x**2) + eps)
```

`weight` is optional, 1-D with the size of the last dimension; `eps >= 0`.
Error vs float64: `|err| ≤ 1e-30 + 1e-6·|ref|` (tests and `selftest`). A NaN in
a row makes the row NaN; an infinity makes its row 0, except NaN at the
infinity (as PyTorch). One pass for Σx² (float lanes added into a double every
1024 elements), one to scale (`docs/DESIGN.md` §3.6).

```python
x = np.random.randn(4, 2048).astype(np.float32)
w = np.ones(2048, np.float32)
y = simd_ml_kernels.rms_norm(x, w)
```

### `silu_mul(gate, up) → np.ndarray`

`SiLU(gate) * up` elementwise in one pass: the gated product of a SwiGLU MLP
(Llama: `down(silu(gate(x)) * up(x))`). `gate` and `up` are float32 arrays of
the same shape. Error vs float64: `|err| ≤ 1e-35·|up| + 1.1e-6·|ref|` for finite
inputs; the result is bitwise `silu(gate) * up` on the same table (SiLU as
`silu`, then one float32 multiply), without the intermediate array
(`docs/DESIGN.md` §3.3).

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
| `sgemm_blocking` | `{kc, mc, nc}` block sizes derived from this core's caches |
| `u8s8_path` | the path `sgemm_u8s8` runs: `avx512_vnni`, `avx_vnni`, `fp32_convert` or `scalar` |
| `q8_0_path` | the path `gemv_q8_0` and `gemv_q4_0` run: `avx512_vnni`, `avx_vnni`, `avx2` or `scalar` |
| `openmp` | whether GEMM can use more than one thread |
| `alignment` | allocation alignment of internal buffers |

### `detected_isa() → str`

The kernel table every call uses: `'avx512'`, `'avx2'` or `'scalar'`; chosen
once, on first use (docs/DESIGN.md §6). `sgemm(..., isa=...)` does not change
it.

### `avx512_available() → bool`

`True` if this build includes the AVX-512 kernels and this CPU and OS can run
them (AVX-512F/DQ/BW/VL and OS ZMM state).

### `f16c_available() → bool`

`True` if this CPU has F16C (`vcvtph2ps`), which the AVX2 kernels require.
`sgemm_f16` also runs without it (scalar kernels).

### `bf16_avx512bf16_available() → bool`

`True` if this CPU has AVX512_BF16 (`vdpbf16ps`). Not required for
`sgemm_bf16`; the `vdpbf16ps` path is used only where this is `True` and
`SIMD_ML_BF16_DOT=1` is set (see `GEMMConfig` above).

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
