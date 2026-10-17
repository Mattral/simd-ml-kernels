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
```

**Implementation**: Goto/BLIS 5-loop structure with panel packing (8×8 register tile, AVX2 FMA). See `docs/DESIGN.md §2–3`.

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

`isa` must be one of `""`, `"avx2"`, `"avx512"`, `"scalar"` — an invalid
value raises `ValueError` at construction time (`GEMMConfig`) or
`ArgumentError` (a `ValueError`; `sgemm()`, checked on every call). The `isa=` parameter is also accepted
directly by `sgemm()`:

```python
C = simd_ml_kernels.sgemm(A, B, isa="avx2")     # forces the 8×8 AVX2 kernel
C = simd_ml_kernels.sgemm(A, B, isa="avx512")   # forces the 8×16 AVX-512 kernel
C = simd_ml_kernels.sgemm(A, B)                 # "" = auto-detect (default)
```

**`isa=` genuinely forces a specific kernel** — this is a real per-call
override, not just a validated-and-ignored hint. Useful for benchmarking
AVX2 vs AVX-512 on the same AVX-512-capable machine, or for reproducing a
specific numeric result independent of what hardware auto-detection would
otherwise choose. The override applies **only to that call** — it does not
change behavior for subsequent calls that omit `isa=`.

Two important edge cases:

- **`isa="avx512"` on hardware without AVX-512F+DQ raises `RuntimeError`.**
  Check `simd_ml_kernels.avx512_available()` first if you're not sure:
  ```python
  if simd_ml_kernels.avx512_available():
      C = simd_ml_kernels.sgemm(A, B, isa="avx512")
  ```
- **`isa="scalar"` raises `RuntimeError` ("not yet implemented"), not a
  silent fallback.** No forced full-matrix scalar path exists for
  `sgemm_packed` yet — scalar code only runs internally for small edge/tail
  blocks that don't fill a complete register tile. Silently falling back to
  SIMD when the user explicitly asked for scalar would be a transparency
  violation, so this is a hard error instead.

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
Implemented as `x · 0.5·(1 + tanh(x/2))` using the Cody–Waite fast tanh.  
Max absolute error: < 2×10⁻⁷ vs float32 reference.

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

### `layer_norm(x, gamma=None, beta=None, eps=1e-5) → np.ndarray`

Layer normalization over the last axis of `x`:

```
output = (x - mean) / sqrt(var + eps) * gamma + beta
```

`gamma` and `beta` are optional learnable affine parameters with shape matching the last dimension of `x`.

```python
x     = np.random.randn(16, 768).astype(np.float32)
gamma = np.ones(768,  dtype=np.float32)   # scale
beta  = np.zeros(768, dtype=np.float32)   # shift

y = simd_ml_kernels.layer_norm(x, gamma=gamma, beta=beta)
# y has zero mean and unit std per row (before affine)
```

**Implementation**: 3-pass AVX2 (mean → variance → normalize) with double-precision horizontal accumulation to prevent catastrophic cancellation.

---

## Threading

### `set_num_threads(n: int) → None`

Set the number of OpenMP threads used by `sgemm`. No-op when built without `-DSIMD_ML_OPENMP`.

### `get_num_threads() → int`

Return the current OpenMP thread count (always 1 without OpenMP).

---

## Diagnostics

### `build_info() → dict`

Return build metadata as a Python dictionary.

```python
info = simd_ml_kernels.build_info()
# Example output:
# {
#   'isa_compiled': 'AVX2+FMA3',
#   'fma': True,
#   'alignment': '64-byte (C++17 aligned operator new)',
#   'build_timestamp': 'Jun 20 2026 14:22:11',
#   'runtime_isa': 'avx2'
# }
```

### `detected_isa() → str`

Return the runtime-detected ISA string: `'avx512'`, `'avx2'`, `'sse42'`, or `'scalar'`. This reflects the process-wide default (computed once at startup) — it does **not** change based on any `isa=` override used on individual `sgemm()` calls.

### `avx512_available() → bool`

Return `True` if this CPU supports AVX-512F+DQ (the instructions required by `sgemm_packed`'s 8×16 micro-kernel), independent of any `isa=` override. Check this before requesting `isa="avx512"` explicitly to avoid a `RuntimeError` on unsupported hardware:

```python
if simd_ml_kernels.avx512_available():
    C = simd_ml_kernels.sgemm(A, B, isa="avx512")
```

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
| requested ISA not available on this CPU (`isa="avx512"`, `sgemm_f16` without F16C), `isa="scalar"` (not implemented) | `RuntimeError` | — |

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
