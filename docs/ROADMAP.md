# simd-ml-kernels — Roadmap

Milestones are numbered work stages, not package versions. Package versions
and their contents are in [CHANGELOG.md](https://github.com/Mattral/simd-ml-kernels/blob/main/CHANGELOG.md).

## Legend
✅ Complete and verified   ⚠️ Partial / in progress   ❌ Not started

---

## Milestone 1 — Correctness & Measurement Foundation ✅
- [✅] Precision test targets compiled without `-ffast-math`
- [✅] Buffer contiguity and dtype validation in Python bindings
- [✅] `tests/test_bindings_edge_cases.py` — rejects non-contiguous NumPy arrays
- [✅] TSC frequency calibration in `main_bench.cpp`
- [✅] CPUID/LFENCE-serialized RDTSC (Intel SDM §4.3)
- [✅] CPU affinity pinning (`sched_setaffinity` on Linux)
- [✅] Real GFLOPS output (utilisation % removed until peak FMA throughput is measured)

---

## Milestone 2 — True GEMM Microkernel ✅
- [✅] B panel packing (`pack_b_panel`)
- [✅] A panel packing (`pack_a_panel`)
- [✅] AVX2 8×8 micro-kernel with 8 YMM accumulators (`inner_kernel_8x8`). Replaced in W3 by 6×16 (12 accumulators), chosen by measurement (DESIGN.md §2.5).
- [✅] 4× K-loop unrolling in micro-kernel + software prefetch
- [✅] Cache-derived tile constants with L1/L2 derivation comments
- [✅] BLIS-style 5-loop structure (jc / pc / ic / jr / ir)
- [✅] Panel packing buffers allocated once per GEMM call (outside k-loop)
- [✅] Second GEMM implementation (`avx_matmul.cpp` — 6×16 register block). Removed in W3: its shape now runs in the one packed GEMM.

---

## Milestone 3 — Activation Kernels ✅
- [✅] GeLU: exact erf (default) and tanh variants, `approximate=` as in PyTorch; max abs error ≤ 2e-6 for both (measured 9.5e-7 / 1.5e-6, see `docs/api/python_api.md`). An earlier entry here said "< 2e-7"; the measured error of the tanh variant was 1.2e-6.
- [✅] ReLU (`_mm256_max_ps`, exact)
- [✅] SiLU/Swish, `x / (1 + exp(−x))` since W3, `|err| ≤ 1e-35 + 1e-6·|ref|` for every finite x. An earlier entry here said "via fast tanh, < 2e-7 error"; that form's error grew without bound below x = −16 (DESIGN.md §3.3).
- [✅] Softmax (numerically stable, max-subtraction, double accumulation)
- [✅] LayerNorm: two passes for the statistics (float lanes added into a double every 1024 elements) and one to normalize, optional γ/β (DESIGN.md §3.5)
- [✅] Shared `simd_math.hpp`: `fast_exp_avx2` + `tanh_avx2`

---

## Milestone 4 — Python Integration & Developer Experience ✅
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

## Milestone 5 — Statistical Benchmarking ✅
- [✅] `bench_stat.cpp`: wall-clock timing, 95% CI, JSON output, CPU pinning
- [✅] Benchmark protocol: `python -m simd_ml_kernels bench --json`, committed results, generated tables (docs/BENCHMARKS.md). The earlier `check_regression.py` gate on shared CI runners was removed.
- [✅] `bench.yml` weekly run on a shared runner, uploaded as an artifact (no regression gate; not used for claims)
- [✅] `docs/BENCHMARKS.md`: protocol, reproduction commands and generated tables (GEMV rows as % of measured bandwidth roofs)
- [✅] OpenBLAS baseline in the same process: NumPy's OpenBLAS in `python -m simd_ml_kernels bench`;
      `-DBENCH_OPENBLAS=ON` links the system BLAS into `bench_stat` (development only)

---

## Milestone 6 — Engineering Polish ✅
- [✅] ODR violation resolved: the test binary links one kernel library. (An earlier entry here said it was resolved by excluding avx_matmul.cpp from a second, non-fast-math "precision" library; that library still duplicated every other kernel and was never actually linked in, so tests ran the -ffast-math build. Removed in favour of linking only the shipped library.)
- [✅] AVX-512 dispatch bug fixed (inner_NR=32 misaligned with 6×16 kernel)
- [✅] All build warnings addressed
- [✅] `guardrail.yml`: clang-tidy, pip-audit, codespell, license check
- [✅] `build-and-test.yml`: GCC-12, Clang-15 ASan, macOS ARM scalar path
- [✅] `docs/DESIGN.md` v2: architecture decisions for all kernels

---

## Milestone 7 — AVX-512 Full Kernel ✅

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
tests (now in `tests/test_gemm_packed.cpp`) that choose
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
shared sandbox with no committed metadata and have been removed. The AVX-512
kernels have since been measured on a 2-vCPU Emerald Rapids cloud VM
(`cloud-xeon-emr-2vcpu_*.json`; FP32, FP16/BF16 and INT8 GEMM and the
activations); see docs/BENCHMARKS.md.

**Remaining scope (not yet done):**
- [✅] Block sizes (KC/MC/NC) are derived at run time from CPUID leaf 4 of the
      running core since W3 (DESIGN.md §2.3), not fixed for Skylake-class caches.
- [❌] Micro-kernel and VNNI tile tuning on an AVX-512 machine: the 12×32
      shapes were chosen from register and port counts and checked under SDE;
      the cloud VM measurements were not used to tune them.
- [✅] AVX-512 BF16/VNNI paths (W4): BF16 `vdpbf16ps` (opt-in) and INT8
      AVX512_VNNI, CPUID-gated, correct under SDE `-spr` / `-icx`; measured on
      the Emerald Rapids VM (DESIGN.md §4).

**Since replaced:** `isa=` (in `sgemm()` / `GEMMConfig`) runs the named
kernel table (`"scalar"`, `"avx2"`, `"avx512"`) for that call; a table this
build or CPU cannot run raises `RuntimeError` (DESIGN.md §2.6, §6). It used
to be a thread-local override inside the AVX2 GEMM file, under which
`isa="scalar"` raised "not yet implemented"; the per-ISA kernel tables of W2
replaced both.

---

## Milestone 8 — Additional Primitives (Planned)
- [✅] Skip panel packing for small products. Since W3 the 6×16 micro-kernel
      runs on A and B in place for products up to 80³, for N ≤ 16 and for
      M = 1 (`sgemm_avx2_prefers_direct`, DESIGN.md §2.1), and the packing
      buffers persist per thread. (The first version, a loop with one FMA
      chain per output strip below `max(M,N,K) ≤ 128`, was latency-bound: 24%
      of OpenBLAS at N = 128.) Tested in `tests/test_gemm_packed.cpp`.
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

- **W2 — portable builds** (done: #17, #18, #19). Per-ISA translation units with explicit flags
  (baseline scalar, AVX2+FMA+F16C, AVX-512F/DQ/BW/VL) instead of
  `-march=native` (kept as an opt-in for local benchmarking); runtime dispatch
  from CPUID + XGETBV, resolved once; a scalar path for every kernel;
  `-ffast-math` only where tests show it is safe; FTZ/DAZ test; AVX-512
  correctness under Intel SDE; wheels built with cibuildwheel, published
  to PyPI from 0.3.0.
- **W3 — GEMM.** Done: one register-blocked 6×16 AVX2 kernel for small
  (unpacked) and large (packed) products, blocking from CPUID leaf 4,
  persistent packing buffers (N = 128 went from 24% to 95% of OpenBLAS on the
  reference laptop; BENCHMARKS.md); AVX-512 12×32 micro-kernel with 24
  accumulators (correct under SDE; 87–107% of OpenBLAS at N = 128–2048 on the
  Emerald Rapids cloud VM, `cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31`, so
  below the 90% target at N = 2048); `trans_a`/`trans_b` in packing;
  LayerNorm two-pass kept (single-pass Welford measured slower for rows up to
  1024 and less accurate); OpenMP threading as a build option (#24).
  Remaining: AVX-512 on bare metal; a thread-scaling table on the i9-9900
  workstation, which decides whether OpenMP becomes the default.
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
  families lost to PyTorch (FP16/BF16 with W [N, K]: W5 item 1).
- **W5 — CPU LLM decode kernels**, judged by achieved GB/s as a % of the
  measured single-core read stream (and STREAM triad) against PyTorch CPU and
  llama.cpp, one PR per kernel family, in this order:
  1. *Bandwidth roofs and the FP32/FP16/BF16 decode GEMV* (done, #32): read
     stream and STREAM triad in `bench`; `x @ W.T` with W [N, K] streamed in
     place for up to 8 rows (`trans_b` for `sgemm_f16`/`sgemm_bf16`);
     DESIGN.md §2.7. On the reference laptop's P-core: FP32 at 97–99% of the
     read stream (1.03–1.05× PyTorch, 1.00–1.20× OpenBLAS sgemv), FP16/BF16
     at 90–99% (1.78–1.92× PyTorch's `F.linear`)
     (`lnl-226v-wsl_2026-10-18_ecc219baa1f2`). AVX-512 speed: not measured;
     the cloud VM's single-core bandwidth varied 5.8–11 GB/s between runs
     (it moved hosts), so a roof measured once per run is not usable there
     (DESIGN.md §2.7).
  2. *Q8_0 GEMV* (done, W5 PR2): `quantize_q8_0` / `gemv_q8_0` (ggml
     `block_q8_0`, byte-identical; x quantized once per call; `vpdpbusd`,
     `vpmaddubsw`), `sgemm_u8s8`'s single row read in place, the read roof
     measured in each case (fastest read-only loop, including the GEMV row
     pattern; `roof_unstable` above 105%). Laptop (`lnl-226v-wsl_2026-10-19_290b91baad26`): 92–95% of the
     roof, 1.46–1.82× ggml's Q8_0 MUL_MAT on one thread, 3.49–3.61× our FP32
     GEMV; `sgemm_u8s8` 1×4096×4096 at 1.36× PyTorch (was 0.46×). DESIGN.md
     §2.8. Open: FP32 GEMV 1×4096×11008 still above 105% of its roof (109.6%
     in that run, not shown): the roof is a lower bound, and a tighter one
     (or a protocol for it) is a later item; AVX-512 GEMV run.
  3. Q4_0 (`quantize_q4_0`, `gemv_q4_0`; ggml `block_q4_0`, rounding as
     `quantize_row_q4_0_ref`).
  4. Fused RMSNorm and SwiGLU (`silu(gate) * up`) epilogues.
  5. Flash-decoding attention for one query (online softmax, GQA, split-K).
  6. Opt-in row-split OpenMP GEMV, judged against the all-core roofs.
- **W6 — end-to-end decode loop** against PyTorch and llama.cpp.

---

## Milestone 9 — NUMA and Multi-socket (Planned)
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
between simd-ml-kernels and production BLAS libraries. Two rows were removed
when W3 fixed them: fixed Skylake-class block sizes (now from CPUID leaf 4)
and small products at 24% of OpenBLAS (93% at N = 128 in
`lnl-226v-wsl_2026-10-18_ecc219baa1f2`).

| Limitation | Impact | Planned fix |
|---|---|---|
| FP32 GEMM below OpenBLAS on large AVX-512 products | 87% of OpenBLAS at N = 2048 and 91% at N = 1024 on the Emerald Rapids cloud VM (`cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31`); the 12×32 tile was not tuned there, and no bare-metal AVX-512 result exists | AVX-512 tuning on a controlled machine |
| Single-threaded default | OpenMP is a build option, off by default; the threaded GEMM (shared B panel, dynamic chunks) has no published scaling numbers yet (needs a non-hybrid 8-core CPU) | W3 |
| INT8 GEMM: single-threaded | No threaded path; AVX512_VNNI on Emerald Rapids: 0.66–0.70× PyTorch at N = 512–2048 (`cloud-xeon-emr-2vcpu_2026-10-18_9029f1566ddb`). One row times a weight matrix now reads it in place (1.36× PyTorch at 1×4096×4096 on the laptop, `lnl-226v-wsl_2026-10-19_290b91baad26`; was 0.46×) | AVX-512 VNNI tile tuning |
| BF16/FP16 GEMM: memory savings only on AVX2 | No 16-bit FMA on AVX2: about the FP32 GEMM's speed on compute-bound shapes; a single row times B stored [K, N] is faster than our FP32 but loses to PyTorch (0.58–0.63×; with W stored [N, K], the decode GEMV, 1.78–1.92× PyTorch, `lnl-226v-wsl_2026-10-18_ecc219baa1f2`); narrow N ≤ 16 shapes pay for converting A (DESIGN.md §4); `vdpbf16ps` measured at 64–73% of our FP32 GEMM on Emerald Rapids, slower than the conversion path (`cloud-xeon-emr-2vcpu_2026-10-18_9029f1566ddb`) | [K, N] single row: not planned (models store W [N, K]); `vdpbf16ps` is opt-in (`SIMD_ML_BF16_DOT=1`) |
| Decode GEMV: one core | Single-threaded, so it is bounded by one core's read bandwidth (32–36 GB/s read roof on the reference laptop), not by DRAM; AVX-512 speed not measured (the cloud VM's bandwidth varied 5.8–11 GB/s between runs); FP32 is level with PyTorch and OpenBLAS (all near that roof); 9–16 rows with W [N, K] now pack W (at most 9% slower than the dot path in dev runs, not in a JSON; DESIGN.md §2.7) | W5 item 6 (opt-in OpenMP GEMV); an AVX-512 run with the per-case roof |
| Read roof is a lower bound | The roof is the fastest read-only loop we have (part sums and the GEMV row pattern); the FP32 GEMV still read 109.6% of it at 1×4096×11008 (`lnl-226v-wsl_2026-10-19_290b91baad26`), so that case is `roof_unstable` and not shown | A tighter single-core read roof, or a protocol for it (after W5 PR2) |
| BF16 GEMM against PyTorch on AMX CPUs | 0.14–0.72× PyTorch's bfloat16 `matmul` on the Emerald Rapids VM (`cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31`); PyTorch's rate implies AMX tiles, which this library does not use (DESIGN.md §4.1) | not planned |
| x86 only, no Windows wheels | SIMD kernels are x86-64 only (scalar elsewhere; macOS arm64 wheels are scalar-only); wheels exist for manylinux x86-64 and macOS arm64, CPython 3.9–3.13; elsewhere pip builds the sdist, which needs a C++17 compiler (pip fetches CMake and Ninja) | not planned |
| No NUMA-aware allocation | Throughput degrades on multi-socket systems | Milestone 9 |
