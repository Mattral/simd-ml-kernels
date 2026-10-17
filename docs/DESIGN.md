# simd-ml-kernels — Design Document

**Version**: 2.0  
**Status**: Living document — updated as major architectural decisions are made.

---

## 1. Project Goals

simd-ml-kernels occupies a specific niche: a **pedagogically transparent yet
engineering-complete** reference implementation of SIMD microkernels for ML.

| Axis               | What we are                         | What we are not |
|--------------------|-------------------------------------|-----------------|
| Correctness        | Validated against NumPy/PyTorch     | A toy / untested sketch |
| Performance        | Measured against OpenBLAS / PyTorch on named hardware (docs/BENCHMARKS.md) | A BLAS replacement |
| Readability        | Commented intrinsics with rationale | Opaque assembly |
| Usability          | `pip install`-able Python library   | An isolated demo |

---

## 2. Cache Blocking (Goto/BLIS Structure)

### Why blocking matters

A naïve triple-loop GEMM (i, j, k) accesses B in column-major order — one
new cache line per inner iteration. For a 256×256 float32 matrix (256 KB),
this causes L2/L3 thrashing and achieves < 5% of peak arithmetic throughput.

### The five-loop structure

We follow the Goto (2008) / BLIS decomposition:

```
for jc in 0..N step NC:          # fits Bc panel in L3
  for pc in 0..K step KC:        # fits Bc panel in L2; Ac panel in L2
    pack_B(B[pc:pc+KC, jc:jc+NC])
    for ic in 0..M step MC:      # fits Ac panel in L2
      pack_A(A[ic:ic+MC, pc:pc+KC])
      for jr in 0..NC step NR:   # register tile N-loop
        for ir in 0..MC step MR: # register tile M-loop
          microkernel(A_packed, B_packed, C)  ← hot inner loop
```

### Tile size rationale

| Constant | Value | Rationale |
|----------|-------|-----------|
| MR       | 8     | 8 YMM accumulators per row; one YMM broadcast per k-step |
| NR       | 8     | 8 floats per YMM; 8 YMM accumulators per col |
| KC       | 256   | Ac panel (MC×KC = 128×256×4B = 128 KB) fits in L2 |
| MC       | 128   | Bc panel (KC×NC = 256×2048×4B = 2 MB) — see NC note |
| NC       | 2048  | Large NC exploits L3; NC is deliberately large in avx2_gemm_packed |

The two GEMM implementations (`avx_matmul.cpp` vs `avx2_gemm_packed.cpp`) use
slightly different tile sizes, providing a useful comparison point. Both follow
the same five-loop structure.

---

## 3. AVX2 Register Blocking

### avx_matmul.cpp: 6×16 register block

```
MR=6, NR=16 (two __m256 vectors wide)
Accumulator registers: 12 YMM (6 rows × 2 columns)
A broadcast:           1  YMM (set1_ps)
B panel:               2  YMM
Total:                15 / 16 YMM — leaves 1 for temporaries
```

### avx2_gemm_packed.cpp: 8×8 register block

```
MR=8, NR=8 (one __m256 vector wide)
Accumulator registers: 8 YMM
A pointers:            8 scalar  (a0..a7[p])
B panel:               1 YMM
Total:                 9 + 8 scalar
```

The 6×16 layout in `avx_matmul.cpp` achieves better arithmetic intensity
(12 FMAs per B-load versus 8) at the cost of a wider B panel.

---

## 4. GeLU Kernel Design

### Mathematical form

Two variants, matching `torch.nn.functional.gelu`:

```
approximate="none" (default):  GeLU(x) = 0.5 · x · [1 + erf(x/√2)]
approximate="tanh":            GeLU(x) ≈ 0.5 · x · [1 + tanh(√(2/π) · (x + 0.044715·x³))]
```

The exact path evaluates erf as a rational minimax approximation
`x·P(x²)/Q(x²)` (deg 6 / deg 4 in x², XLA `ErfImpl32` coefficients) on x clamped
to [−4, 4]: 11 FMA/MUL and one VDIVPS per 8 values, no table lookups. Measured
error bounds are in `docs/api/python_api.md`. The relative throughput of the two
variants is not measured yet; it is added with the benchmark protocol (ROADMAP).
`gelu_forward_scalar` (libm `erff`) remains the scalar correctness reference.

### Rational polynomial for tanh

```
tanh(y) ≈ y · (c₁ + c₃·y² + c₅·y⁴)
           ────────────────────────────
           1  + d₂·y² + d₄·y⁴ + d₆·y⁶
```

Coefficients (minimax fit over [−5, 5]):

| Coefficient | Value       |
|-------------|-------------|
| c₃          | −0.16035530 |
| c₅          |  0.00533740 |
| d₂          |  0.48057120 |
| d₄          |  0.07985040 |
| d₆          |  0.00587330 |

Division is replaced by `_mm256_rcp_ps` (≈ 12 bits) + one Newton–Raphson
refinement step, reaching full FP32 accuracy (~23 bits) at 1.0 cycle
throughput vs 5–9 cycles for true division.

**Measured:** GeLU(tanh) is within 1.5e-6 of the tanh formula in float64 on [−10, 10] (bound 2e-6, asserted in tests).

---

## 5. Layer Normalization Design

Layer Normalization (Ba et al., 2016) normalizes over the last dimension:

```
output_i = (x_i − μ) / √(σ² + ε) · γ_i + β_i
```

### Three-pass implementation

1. **Mean pass**: AVX2 horizontal sum with float→double promotion to
   avoid catastrophic cancellation.
2. **Variance pass**: AVX2 vectorized (x − mean)² accumulation.
3. **Normalize pass**: AVX2 vectorized multiply + optional gamma/beta FMA.

The three-pass approach is chosen over Welford's online algorithm for
pedagogical clarity; it produces simpler, easier-to-verify assembly.
For very large n (>10⁶), a two-pass (combined variance+normalize) would
reduce memory traffic — noted in ROADMAP.md.

---

## 6. Software Prefetching

The GEMM microkernel issues software prefetch hints for the B panel:

```cpp
prefetch_l1(b_ptr + 32);   // 4 iterations ahead in the K-loop
```

This is a hint to pull the next cache line into L1 before it is needed.
On Skylake-class CPUs, L2→L1 latency is ~12 cycles; an FMA throughput of
0.5 cy/instruction means the kernel processes ~24 FMAs in that window —
matching the 4-iteration unroll factor.

**Prefetch distance tuning**: the constant `+32` (2 YMM vectors, 64 bytes,
one cache line) is appropriate for Skylake. Zen 3/4 or Alder Lake may
benefit from a larger or smaller distance. This is a known tuning opportunity.

---

## 7. Explicitly Acknowledged Gaps vs Production BLAS

| Gap                          | Impact (estimated)     | Future work |
|------------------------------|------------------------|-------------|
| C++ intrinsics vs assembly   | 5–15% overhead         | Optional; use libxsmm? |
| Fixed tile sizes             | 10–20% suboptimal      | Empirical search / BLIS auto-tune |
| No NUMA-aware allocation     | Negligible for 1 socket| Add for multi-socket |
| No packing-skip for small N  | ~~GEMM slower than scalar below N≈128~~ **Fixed in v0.9**: `sgemm_direct_avx2` (plain AVX2 loop, no packing) dispatched automatically for `max(M,N,K) ≤ 128` | Done |
| Single-threaded default      | Linear with core count | OpenMP (optional flag) |

---

## 8. `-ffast-math` Policy

The performance build uses `-ffast-math`. This permits:
- Reassociation of FP operations (may change rounding order)
- Unsafe math optimizations (NaN/Inf behaviour undefined)

This is **appropriate for ML inference** where:
- Inputs are pre-validated (no NaN/Inf expected)
- Sub-ULP rounding differences do not affect model accuracy
- The speed gain (5–15%) is material

The precision test library is built **without** `-ffast-math` to validate
correctness under IEEE 754 semantics.

---

## 9. Python Binding Design

Key decisions in `pybind_entry.cpp`:

1. **GIL release**: `py::gil_scoped_release release;` before every kernel
   call. This allows Python's threading module to schedule other threads
   during computation, important for async inference servers.

2. **Zero-copy**: `py::array_t<float, py::array::c_contiguous>` gives
   direct pointer access to the NumPy buffer — no copies.

3. **Shape validation at the Python layer**: clearer errors than catching
   undefined behaviour inside the kernel.

4. **C allocation on None**: `sgemm(A, B)` without a C argument allocates
   a new output array, matching the NumPy `@` operator ergonomics.

---

## 10. CI/CD Architecture

Four GitHub Actions workflows:

| Workflow          | Trigger                | Purpose |
|-------------------|------------------------|---------|
| `ci.yml`          | Every push / PR        | Build + C++ tests + Python precision tests |
| `build-and-test.yml` | Push to main        | Focused build validation |
| `bench.yml`       | Weekly (Sunday 4 AM)   | Statistical benchmark + regression gate |
| `guardrail.yml`   | Every push / PR        | Static analysis + security scan |

The bench workflow uses `bench_stat` (not `bench`) for statistical rigour:
30 measurement repetitions, 95% confidence intervals, JSON output, and
committed, metadata-complete results (`python -m simd_ml_kernels bench --json`).

---

*References*

- Goto, K. & van de Geijn, R. (2008). Anatomy of High-Performance Matrix Multiplication. *ACM TOMS* 34(3).
- Van Zee, F.G. & van de Geijn, R. (2015). BLIS: A Framework for Rapidly Instantiating BLAS Functionality. *ACM TOMS* 41(3).
- Hendrycks, D. & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv:1606.08415*.
- Ba, J.L. et al. (2016). Layer Normalization. *arXiv:1607.06450*.
- Intel. *Intrinsics Guide*. https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
