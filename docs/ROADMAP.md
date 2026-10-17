# IntrinsicML — Roadmap

## Legend
✅ Complete and verified   ⚠️ Partial / in progress   ❌ Not started

---

## v0.2 — Correctness & Measurement Foundation ✅
- [✅] Precision test targets compiled without `-ffast-math`
- [✅] Buffer contiguity and dtype validation in Python bindings
- [✅] `tests/test_bindings_edge_cases.py` — rejects non-contiguous NumPy arrays
- [✅] TSC frequency calibration in `main_bench.cpp`
- [✅] CPUID/LFENCE-serialized RDTSC (Intel SDM §4.3)
- [✅] CPU affinity pinning (`sched_setaffinity` on Linux)
- [✅] Real GFLOPS output with hardware utilisation %

---

## v0.3 — True GEMM Microkernel ✅
- [✅] B panel packing (`pack_b_panel`)
- [✅] A panel packing (`pack_a_panel`)
- [✅] AVX2 8×8 micro-kernel with 8 YMM accumulators (`inner_kernel_8x8`)
- [✅] 4× K-loop unrolling in micro-kernel + software prefetch
- [✅] Cache-derived tile constants with L1/L2 derivation comments
- [✅] BLIS-style 5-loop structure (jc / pc / ic / jr / ir)
- [✅] Panel packing buffers allocated once per GEMM call (outside k-loop)
- [✅] Second GEMM implementation (`avx_matmul.cpp` — 6×16 register block)

---

## v0.4 — Activation Kernels ✅
- [✅] GeLU (Cody–Waite exp-based tanh, < 2e-7 absolute error)
- [✅] ReLU (`_mm256_max_ps`, exact)
- [✅] SiLU/Swish (`x·sigmoid(x)` via fast tanh, < 2e-7 error)
- [✅] Softmax (numerically stable, max-subtraction, double accumulation)
- [✅] LayerNorm (3-pass AVX2, double horizontal accumulation, optional γ/β)
- [✅] Shared `simd_math.hpp`: production-grade `fast_exp_avx2` + `tanh_avx2`

---

## v0.5 — Python Integration & Developer Experience ✅
- [✅] pybind11 bindings for all kernels: `sgemm`, `gelu`, `relu`, `silu`, `softmax`, `layer_norm`
- [✅] GIL release during all kernel calls (threading-safe)
- [✅] Python type stubs (`simd_kernels.pyi`) for IDE autocompletion
- [✅] `simd_kernels.build_info()` / `detected_isa()` / `is_aligned()`
- [✅] `set_num_threads` / `get_num_threads` exposed to Python
- [✅] `GEMMConfig` callable configuration object (roadmap §9 DX vision)
- [✅] `isa=` keyword argument on `sgemm()` and `GEMMConfig` (validated, forward-compatible)
- [✅] 100+ Python tests (precision × kernels × shapes, edge cases, error handling)
- [✅] Optional PyTorch cross-validation in test suite

---

## v0.6 — Statistical Benchmarking ✅
- [✅] `bench_stat.cpp`: wall-clock timing, 95% CI, JSON output, CPU pinning
- [✅] `benchmarks/check_regression.py`: CI regression gate (configurable threshold)
- [✅] `bench.yml` weekly cron + automated regression check
- [✅] `docs/BENCHMARKS.md` with roofline analysis and reproducibility instructions
- [✅] OpenBLAS baseline comparison with real measured data (`-DBENCH_OPENBLAS=ON`,
      wired into both `bench` and `bench_stat`; see `docs/BENCHMARKS.md §Positioning vs OpenBLAS`)

---

## v0.7 — Engineering Polish ✅
- [✅] ODR violation resolved: the test binary links one kernel library. (An earlier entry here said it was resolved by excluding avx_matmul.cpp from a second, non-fast-math "precision" library; that library still duplicated every other kernel and was never actually linked in, so tests ran the -ffast-math build. Removed in favour of linking only the shipped library.)
- [✅] AVX-512 dispatch bug fixed (inner_NR=32 misaligned with 6×16 kernel)
- [✅] All build warnings addressed
- [✅] `guardrail.yml`: clang-tidy, pip-audit, codespell, license check
- [✅] `build-and-test.yml`: GCC-12, Clang-15 ASan, macOS ARM scalar path
- [✅] `docs/DESIGN.md` v2: architecture decisions for all kernels

---

## v0.8 — AVX-512 Full Kernel ✅

**Both GEMM kernels now have correct, tested, dispatched AVX-512 support:**

| Kernel | File | AVX2 tile | AVX-512 tile | Status |
|---|---|---|---|---|
| Reference | `avx_matmul.cpp` | 6×16 (12 YMM) | 6×32, dual `__m512`/row (12 ZMM) | ✅ |
| Primary (Python-facing) | `avx2_gemm_packed.cpp` | 8×8 (8 YMM) | 8×16, single `__m512`/row (8 ZMM) | ✅ |

**Design note on the two different AVX-512 tile shapes:** the reference
kernel's 6×32 tile needed a genuine dual-accumulator split (two `__m512`
per row) because 32 columns exceeds one ZMM register's 16-float width —
this is exactly the design the disabled stub got wrong (see below). The
primary kernel's 8×16 tile needed no such split: one `__m512` already
spans the full 16-column width, so there is no "second half" to forget by
construction. Both were verified via standalone hand-computed unit-test
harnesses run on real AVX-512 hardware before being wired into their
respective dispatch paths, plus permanent forced-full-coverage regression
tests in `tests/test_gemm.cpp` and `tests/test_gemm_packed.cpp` that choose
M/N dimensions with zero possible edge/tail blocks — so a broken full-block
kernel cannot be masked by scalar-fallback correctness. Both pass under
AddressSanitizer + UndefinedBehaviorSanitizer with zero findings.

**History — why this was previously disabled:** an earlier version of
`gemm_micro_6x32_avx512` declared a 32-wide tile but only ever loaded and
accumulated one `__m512` (16 floats) per row, silently leaving the upper 16
output columns untouched after a `beta=0` memset (i.e., silently zeroed).
That bug is why AVX-512 dispatch was disabled for an entire release cycle —
`inner_NR` was fixed at 16 (AVX2-only) regardless of detected hardware.

**Measured performance impact (primary Python-facing kernel, this environment):**

| N   | AVX2-only (previous) | AVX-512 (now) | Speedup |
|-----|----------------------|----------------|---------|
| 64  | 5.4 GFLOPS  | 6.4–6.6 GFLOPS  | ~1.15–1.2× |
| 128 | 26.9 GFLOPS | 30–35 GFLOPS    | ~1.1–1.3× |
| 256 | 54.9 GFLOPS | 64–66 GFLOPS    | ~1.15–1.2× |
| 512 | 58.4 GFLOPS | 73–74 GFLOPS    | ~1.25–1.3× |

These numbers were captured on a shared, unpinned CI sandbox with visible
run-to-run scheduler noise (see raw trial data in commit history); repeated
trials cluster consistently in the ranges shown above. **Why not closer to
2×** despite doubling SIMD width: panel packing and memory-movement overhead
are ISA-independent — only the inner FMA loop benefits from the wider
register, so overall speedup is bounded well below 2× by Amdahl's law. Run
`./bench_stat --sizes 64,128,256,512 --reps 50` with `sudo cpupower
frequency-set -g performance` and `taskset -c 0` on dedicated hardware for
tighter, more reproducible numbers.

**Remaining scope (not yet done):**
- [❌] Tile-size re-tuning for Sapphire Rapids / Zen 4 (larger L2, different
      port layout) — current MC/KC/NC constants were derived for Skylake-class
      caches and are not yet re-validated on newer AVX-512 microarchitectures.
- [❌] AVX-512 BF16/VNNI paths (separate from the FP32 work done here).

**Since resolved:** `isa=` (in `sgemm()` / `GEMMConfig`) now genuinely forces
a specific kernel per-call — `set_gemm_isa_override`/`get_gemm_isa_override`
(thread_local, RAII-guarded around each call so it never leaks into
subsequent calls) in `avx2_gemm_packed.cpp`, wired through
`pybind_entry.cpp`. Requesting `isa="avx512"` on hardware without
AVX-512F+DQ raises `RuntimeError` (checked against
`gemm_packed_avx512_hardware_available()`, also exposed to Python as
`simd_kernels.avx512_available()`). `isa="scalar"` is a recognized value
but raises `RuntimeError` ("not yet implemented") rather than silently
falling back, since no forced full-matrix scalar path exists yet. Verified
via a direct C++ test (`test_gemm_isa_override` in
`tests/test_gemm_packed.cpp`) that proves the override changes
`gemm_packed_isa_is_avx512()`'s return value in both directions — not
inferred from numeric equivalence, which could mask a broken override on
hardware where auto-detect and a forced choice happen to coincide.

---

## v0.9 — Additional Primitives (Planned)
- [✅] Skip panel packing below a size threshold (direct unpacked micro-kernel
      call for small N). `sgemm_direct_avx2`: plain AVX2 FMA loop on original
      strides, no buffer allocation. Dispatched automatically from `sgemm_packed`
      when `max(M,N,K) ≤ SMALL_GEMM_THRESHOLD=128`. At N=64, the fixed KC×NC
      packing buffer (≈2 MB regardless of actual N) was costing more than the
      arithmetic itself, giving ~0.7× naive scalar. The direct path fixes this.
      Tested in `test_gemm_packed.cpp::test_small_gemm_direct_path()` across 8
      representative shapes including the key N=64 regression case.
- [✅] BF16 GEMM kernel (AVX2 zero-extend path; vdpbf16ps packing path → v1.0).
      `sgemm_bf16_avx2`: converts BF16→FP32 using only AVX2 integer ops
      (`_mm256_cvtepu16_epi32` + `_mm256_slli_epi32`, 3 instructions per 8
      values — cheaper than F16C's VCVTPH2PS). Requires no special compile
      flag beyond AVX2. `bf16_avx512bf16_available()` provides a runtime
      check for vdpbf16ps availability (Ice Lake-SP+) ahead of the v1.0
      packing path. Exposed as `simd_kernels.sgemm_bf16(A, B)` → float32.
      Tested in `tests/test_gemm_bf16.cpp` across 14 shapes including BF16
      bit-pattern unit test and beta=0 NaN-safety check.
- [✅] FP16 GEMM kernel (F16C extension). `sgemm_f16_avx2`: converts FP16
      inputs on load via `_mm256_cvtph_ps` (1 instruction per 8 values, hidden
      behind FMA latency), accumulates in FP32, stores FP32 output. Exposed as
      `simd_kernels.sgemm_f16(A_f16, B_f16)` → `ndarray[float32]`. Runtime
      guard via `f16c_available()` / `simd_kernels.f16c_available()`.
      Implemented in `src/kernels/gemm/f16_gemm.{hpp,cpp}`, tested in
      `tests/test_gemm_f16.cpp` across 14 shapes including single-row
      inference (M=1, N=64, K=128) and alpha/beta variants.
- [✅] Vectorized `exp` via Cody–Waite (`fast_exp_avx2` in `simd_math.hpp`, shared by GeLU/SiLU/Softmax)
- [✅] Vectorized Softmax inner loop using fast exp approximation (`softmax_avx2.cpp` Pass 2)
- [❌] `FlashAttention`-style fused attention microkernel (research target)

---

## v1.0 — NUMA and Multi-socket (Planned)
- [❌] NUMA-aware packing buffer allocation
- [❌] Multi-socket OpenMP with NUMA-local thread-private buffers
- [❌] Benchmark against OpenBLAS multi-threaded on dual-socket systems

---

## GeLU Benchmark Platform Dependency

On Ubuntu 22.04+ with glibc ≥ 2.22, the compiler auto-vectorises loops
containing `erff()` via the `_ZGVdN8v_erff` SVML symbol. This means the scalar
`gelu_forward_scalar` (erff path) and `gelu_forward_avx2` (Cody–Waite tanh)
run at approximately the same throughput on this platform.

**This is not a bug.** On platforms without SVML (Alpine Linux, older glibc,
macOS, embedded targets), `erff` runs scalar and the AVX2 tanh path achieves
6–10× speedup. The AVX2 tanh path also avoids the `erff` function-call overhead
entirely, producing more predictable latency in inference pipelines.

**What this means for benchmarks:** The GeLU column in BENCHMARKS.md shows ~1×
speedup on CI (Ubuntu 22.04 runner). This is accurate for that environment.
On production servers running older glibc or containers based on Alpine, the
speedup will be in the 6–10× range.

---

## Known Honest Limitations (never remove this section)

These limitations are explicitly documented to help users understand the gap
between IntrinsicML and production BLAS libraries.

| Limitation | Impact | Planned fix |
|---|---|---|
| No hand-written assembly micro-kernel | 5–15% overhead vs hand-tuned assembly | Optional / long-term |
| Fixed tile sizes (MC/KC/NC) | Derived for Skylake-class caches; not re-tuned for Sapphire Rapids / Zen 4 | v0.9 auto-tune |
| Panel packing not skipped for small N | `sgemm_packed` is *slower* than a plain scalar loop at N=64 (~0.7×) — packing overhead dominates before it's amortised | v0.9 |
| Single-threaded default | Linear throughput scaling with cores uncaptured | OpenMP flag exists |
| No BF16/FP16 kernels | Modern LLM inference prefers lower precision | v0.9 |
| No NUMA-aware allocation | Throughput degrades on multi-socket systems | v1.0 |
