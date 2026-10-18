# simd-ml-kernels — Roadmap

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
- [✅] Real GFLOPS output (utilisation % removed until peak FMA throughput is measured)

---

## v0.3 — True GEMM Microkernel ✅
- [✅] B panel packing (`pack_b_panel`)
- [✅] A panel packing (`pack_a_panel`)
- [✅] AVX2 8×8 micro-kernel with 8 YMM accumulators (`inner_kernel_8x8`). Replaced in W3 by 6×16 (12 accumulators), chosen by measurement (DESIGN.md §2.5).
- [✅] 4× K-loop unrolling in micro-kernel + software prefetch
- [✅] Cache-derived tile constants with L1/L2 derivation comments
- [✅] BLIS-style 5-loop structure (jc / pc / ic / jr / ir)
- [✅] Panel packing buffers allocated once per GEMM call (outside k-loop)
- [✅] Second GEMM implementation (`avx_matmul.cpp` — 6×16 register block). Removed in W3: its shape now runs in the one packed GEMM.

---

## v0.4 — Activation Kernels ✅
- [✅] GeLU: exact erf (default) and tanh variants, `approximate=` as in PyTorch; max abs error ≤ 2e-6 for both (measured 9.5e-7 / 1.5e-6, see `docs/api/python_api.md`). An earlier entry here said "< 2e-7"; the measured error of the tanh variant was 1.2e-6.
- [✅] ReLU (`_mm256_max_ps`, exact)
- [✅] SiLU/Swish (`x·sigmoid(x)` via fast tanh, < 2e-7 error)
- [✅] Softmax (numerically stable, max-subtraction, double accumulation)
- [✅] LayerNorm (3-pass AVX2, double horizontal accumulation, optional γ/β)
- [✅] Shared `simd_math.hpp`: production-grade `fast_exp_avx2` + `tanh_avx2`

---

## v0.5 — Python Integration & Developer Experience ✅
- [✅] pybind11 bindings for all kernels: `sgemm`, `gelu`, `relu`, `silu`, `softmax`, `layer_norm`
- [✅] GIL release during all kernel calls (threading-safe)
- [✅] Python type stubs (`python/simd_ml_kernels/_core.pyi`) for IDE autocompletion
- [✅] `simd_ml_kernels.build_info()` / `detected_isa()` / `is_aligned()`
- [✅] `set_num_threads` / `get_num_threads` exposed to Python
- [✅] `GEMMConfig` callable configuration object (roadmap §9 DX vision)
- [✅] `isa=` keyword argument on `sgemm()` and `GEMMConfig` (validated, forward-compatible)
- [✅] 100+ Python tests (precision × kernels × shapes, edge cases, error handling)
- [✅] Optional PyTorch cross-validation in test suite

---

## v0.6 — Statistical Benchmarking ✅
- [✅] `bench_stat.cpp`: wall-clock timing, 95% CI, JSON output, CPU pinning
- [✅] Benchmark protocol: `python -m simd_ml_kernels bench --json`, committed results, generated tables (docs/BENCHMARKS.md). The earlier `check_regression.py` gate on shared CI runners was removed.
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
| Reference (removed in W3) | `avx_matmul.cpp` | 6×16 (12 YMM) | 6×32, dual `__m512`/row (12 ZMM) | removed |
| Primary (Python-facing) | `avx2_gemm_packed.cpp` | 6×16 (12 YMM) since W3; 8×8 before | 12×32, two `__m512`/row (24 ZMM) since W3; 8×16 before | ✅ |

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

Performance numbers that used to be listed here came from an unpinned,
shared sandbox with no committed metadata and have been removed. Neither
machine available now has AVX-512, so the AVX-512 kernels cannot be
re-measured; see docs/BENCHMARKS.md for what is measured.

**Remaining scope (not yet done):**
- [❌] Tile-size re-tuning for Sapphire Rapids / Zen 4 (larger L2, different
      port layout) — current MC/KC/NC constants were derived for Skylake-class
      caches and are not yet re-validated on newer AVX-512 microarchitectures.
- [✅] AVX-512 BF16/VNNI paths (W4): BF16 `vdpbf16ps` and INT8 AVX512_VNNI,
      CPUID-gated, correct under SDE `-spr` / `-icx`; speed not measured.

**Since resolved:** `isa=` (in `sgemm()` / `GEMMConfig`) now genuinely forces
a specific kernel per-call — `set_gemm_isa_override`/`get_gemm_isa_override`
(thread_local, RAII-guarded around each call so it never leaks into
subsequent calls) in `avx2_gemm_packed.cpp`, wired through
`pybind_entry.cpp`. Requesting `isa="avx512"` on hardware without
AVX-512F+DQ raises `RuntimeError` (checked against
`gemm_packed_avx512_hardware_available()`, also exposed to Python as
`simd_ml_kernels.avx512_available()`). `isa="scalar"` is a recognized value
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
      arithmetic itself. The direct path fixes this.
      Tested in `test_gemm_packed.cpp::test_small_gemm_direct_path()` across 8
      representative shapes including the key N=64 regression case.
- [✅] BF16 and FP16 GEMM (`sgemm_bf16`, `sgemm_f16` → float32). Since W4 the
      inputs are converted to FP32 once while packing and the FP32 micro-kernels
      run unchanged (DESIGN.md §4); the first versions converted B on every use
      without blocking and ran at 12–17% of the FP32 GEMM. `vdpbf16ps` on
      AVX512_BF16 CPUs. Tested in `tests/test_gemm_lowp.cpp` and
      `tests/test_precision.py` against float64.
- [✅] Vectorized `exp` via Cody–Waite (`fast_exp_avx2` in `simd_math.hpp`, shared by GeLU/SiLU/Softmax)
- [✅] Vectorized Softmax inner loop using fast exp approximation (`softmax_avx2.cpp` Pass 2)
- [❌] `FlashAttention`-style fused attention microkernel (research target)

---

## Next, in order

Each item is done when its tests, committed benchmark JSON and docs are in.

- **W2 — portable builds** (in review: #17, #18, #19). Per-ISA translation units with explicit flags
  (baseline scalar, AVX2+FMA+F16C, AVX-512F/DQ/BW/VL) instead of
  `-march=native` (kept as an opt-in for local benchmarking); runtime dispatch
  from CPUID + XGETBV, resolved once; a scalar path for every kernel;
  `-ffast-math` only where tests show it is safe; FTZ/DAZ test; AVX-512
  correctness under Intel SDE; wheels built with cibuildwheel (not published).
- **W3 — GEMM.** Done: one register-blocked 6×16 AVX2 kernel for small
  (unpacked) and large (packed) products, blocking from CPUID leaf 4,
  persistent packing buffers (N = 128 went from 24% to 95% of OpenBLAS on the
  reference laptop; BENCHMARKS.md); AVX-512 12×32 micro-kernel with 24
  accumulators (correct under SDE; speed not measured yet); `trans_a`/`trans_b`
  in packing; LayerNorm two-pass kept (single-pass Welford measured slower
  for rows up to 1024 and less accurate). Remaining: AVX-512 measurement on
  real hardware; OpenMP on by default with a scaling table. Measured on
  ≥ 2 CPUs.
- **W4 — low precision that is faster than FP32.** Done: BF16/FP16 converted
  at pack time (95–101% of the FP32 GEMM at N = 128–2048, faster where
  reading B dominates; memory savings only on AVX2, which has no 16-bit FMA;
  BENCHMARKS.md), `vdpbf16ps` behind CPUID (measured on Emerald Rapids: 64–73% of our FP32
  GEMM, slower than the conversion path, so opt-in via `SIMD_ML_BF16_DOT=1`); INT8
  `sgemm_u8s8` with per-channel scales: AVX-VNNI 2.39–3.68× the FP32 GEMM and
  1.66–2.04× PyTorch's quantized linear at N = 128–2048, AVX512_VNNI correct
  under SDE and measured on Emerald Rapids (1.43–2.75× our FP32 GEMM,
  0.66–0.70× PyTorch's quantized linear at N = 512–2048), FP32 conversion path (memory only) without
  VNNI. Left for W5: single-row (GEMV) shapes, where both low-precision
  families lose to PyTorch.
- **W5 — CPU LLM decode kernels.** Quantized GEMV with fused dequantization,
  RMSNorm/SwiGLU epilogues, flash-decoding attention; judged by % of measured
  STREAM bandwidth vs PyTorch CPU and llama.cpp.
- **W6 — end-to-end decode loop** against PyTorch and llama.cpp.

---

## v1.0 — NUMA and Multi-socket (Planned)
- [❌] NUMA-aware packing buffer allocation
- [❌] Multi-socket OpenMP with NUMA-local thread-private buffers
- [❌] Benchmark against OpenBLAS multi-threaded on dual-socket systems

---

## GeLU Benchmark Platform Dependency

(An earlier note here estimated GeLU speedups over a scalar `erff` loop on
various platforms; none of it was measured, and it is removed. GeLU
throughput against PyTorch is in docs/BENCHMARKS.md.)

---

## Known Honest Limitations (never remove this section)

These limitations are explicitly documented to help users understand the gap
between simd-ml-kernels and production BLAS libraries.

| Limitation | Impact | Planned fix |
|---|---|---|
| Fixed tile sizes (MC/KC/NC) | Chosen for Skylake-class caches; not derived from the running CPU | W3 |
| Small matrices (max(M,N,K) ≤ 128) | One FMA chain per output strip: 24% of OpenBLAS at N = 128 on the reference laptop | W3 |
| Single-threaded default | OpenMP is a build option, off by default; the threaded GEMM (shared B panel, dynamic chunks) has no published scaling numbers yet (needs a non-hybrid 8-core CPU) | W3 |
| INT8 GEMM: single-threaded; GEMV shapes | No threaded path; one row times a weight matrix runs at 0.46× PyTorch's quantized linear (every call packs all of B), and the FP32 conversion path (CPUs without VNNI) at 0.47× our FP32 GEMM there (DESIGN.md §4.2); AVX512_VNNI on Emerald Rapids: 0.66–0.70× PyTorch at N = 512–2048 (`cloud-xeon-emr-2vcpu_2026-10-18_9029f1566ddb`) | W5 (quantized GEMV), AVX-512 VNNI tile tuning |
| BF16/FP16 GEMM: memory savings only on AVX2 | No 16-bit FMA on AVX2: about the FP32 GEMM's speed on compute-bound shapes; the single-row case is faster than our FP32 but loses to PyTorch (0.58–0.63×); narrow N ≤ 16 shapes pay for converting A (DESIGN.md §4); `vdpbf16ps` measured at 64–73% of our FP32 GEMM on Emerald Rapids, slower than the conversion path (`cloud-xeon-emr-2vcpu_2026-10-18_9029f1566ddb`) | W5 (GEMV); `vdpbf16ps` is opt-in (`SIMD_ML_BF16_DOT=1`) |
| No NUMA-aware allocation | Throughput degrades on multi-socket systems | v1.0 |
