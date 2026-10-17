# simd-ml-kernels — Design

How the library is put together and why. Measured performance is in
[BENCHMARKS.md](BENCHMARKS.md) (generated from committed JSON); this page gives
derivations and reasons, not results. Commands, the repository map and the
checklist for adding a kernel are in [AGENTS.md](../AGENTS.md).

This file replaces `architecture.md`, `system_design.md` and `design/*.md`,
which described the same code several times and had drifted from it (see the
"Corrections" note at the end).

---

## 1. Layers

```
Python   simd_ml_kernels/__init__.py  (re-exports _core)   _cli.py (info | selftest | bench)
           │
pybind11 src/bindings/pybind_entry.cpp: dtype/shape/layout checks, GIL release, output allocation
           │  every call goes through kernels(), the table chosen for this CPU
Dispatch src/dispatch/: CPUID + XGETBV, one KernelTable per ISA, selection (§6)
           │
Kernels  scalar unit   scalar_kernels.cpp                 every kernel, portable C++
         AVX2 units    gemm/avx2_gemm_packed.cpp           FP32 GEMM: small-matrix path, 8×8 kernel
                       gemm/{f16,bf16}_gemm.cpp            FP16- / BF16-input GEMM (unpacked, see §4)
                       intrinsic_gelu.cpp, activations/    GeLU, ReLU, SiLU, softmax, LayerNorm
                       avx_matmul.cpp                      second FP32 GEMM (6×16), C++ only
         AVX-512 units gemm/avx512_gemm_packed.cpp         FP32 GEMM, 8×16 kernel
                       avx_matmul_avx512.cpp               6×32 kernel of the second GEMM
           │
Shared   simd_math.hpp (fast_exp_avx2, tanh_avx2), cache_alloc.hpp (aligned allocation, prefetch)
```

Every Python entry point validates its arguments (`DTypeError` for dtype,
`ArgumentError` for shape, contiguity, writeability and values), allocates the
output if none was given, releases the GIL and calls one kernel through the
active table. Kernels never throw and never allocate Python objects.

---

## 2. FP32 GEMM (`sgemm`)

`sgemm(A, B, C, alpha, beta)` computes `C = alpha·A·B + beta·C`, row-major,
through the active kernel table (§6).

### 2.1 Small matrices: no packing

If `max(M, N, K) ≤ 128` (`SMALL_GEMM_THRESHOLD`), `sgemm_direct_avx2` runs
instead: for each row of C and each 8-column strip, one YMM accumulator
collects `Σ_p A[i][p]·B[p][j:j+8]` directly from the unpacked inputs. At
N = 128, A + B + C are 192 KiB, inside the L2 of any recent core, so strided B
reads hit in cache. The K loop of one strip is a single chain of dependent
FMAs (one accumulator; GCC 13 emits one `vfmadd231ps` per iteration), so one
strip alone runs at one FMA per 4-cycle latency, 4 of the 32 flop/cycle an
AVX2 core can do. Only the out-of-order window overlaps neighbouring strips:
the reference laptop measures 6.1 flop/cycle at N = 128, about a quarter of
OpenBLAS (BENCHMARKS.md). A register-blocked kernel without packing is the
first target of W3 (ROADMAP).

### 2.2 Large matrices: Goto/BLIS 5 loops

```
for jc in 0..N step NC:            B panel  KC×NC, packed once per (jc, pc)
  for pc in 0..K step KC:
    pack_b_panel(B[pc:pc+KC, jc:jc+NC])     → NR-wide column blocks
    for ic in 0..M step MC:
      pack_a_panel(A[ic:ic+MC, pc:pc+KC])   → row-major MC×KC
      for jr in 0..NC step NR:              micro-panel of B: KC×NR
        for ir in 0..MC step MR:
          micro-kernel: C[ir:ir+MR, jr:jr+NR] += alpha · A_packed · B_packed
```

`beta` is applied to all of C once before the loops (`beta = 0` writes zeros
and never reads C, so NaN in an uninitialised C does not propagate). Tail
blocks (`mr < MR` or `nr < NR`) go through a scalar loop over the packed
panels. Both packing buffers are allocated once per call; with OpenMP each
thread has its own (`thread_local`) and the `jc` loop is parallel.

Packed layouts:

```
B_packed[(j / NR) * kc * NR + p * NR + (j % NR)] = B[p][j]   (zero-padded to NR)
A_packed[i * kc + p]                              = A[i][p]
```

### 2.3 Blocking constants

| Constant | Value | Footprint | Intended level |
|---|---:|---|---|
| MR × NR | 8 × 8 (AVX2), 8 × 16 (AVX-512) | 8 accumulator registers | registers |
| KC × NR | 256 × 8 | 8 KiB (16 KiB for NR = 16) B micro-panel, reused for all MC/MR row blocks | L1 |
| MC × KC | 128 × 256 | 128 KiB packed A | L2 |
| KC × NC | 256 × 2048 | 2 MiB packed B | L3 (or a large L2) |

They are compile-time constants chosen for a Skylake-class core (32 KiB L1D,
256 KiB–1.25 MiB L2). On the reference laptop's Lion Cove P-core (48 KiB L1D,
2.5 MiB L2, from CPUID leaf 4) the A panel uses 5% of L2. Deriving them from
CPUID leaf 4 at run time is W3. `GEMMConfig(tile_m=..., ...)` accepts tile
sizes but does not use them.

### 2.4 Micro-kernels

**8×8 AVX2** (`inner_kernel_8x8`). Per k-step: one 8-float B load, 8
broadcasts of `A[i][p]` (`vbroadcastss` from memory, a load-port µop each) and
8 FMAs into 8 independent YMM accumulators. K is unrolled by 4 with one
`prefetcht0` per unrolled step, 4 k-steps (128 bytes) ahead in B; those 4
k-steps read two cache lines of B, so software prefetches every second line
and the hardware prefetcher has to fetch the others.

- Latency: an FMA has 4 cycles latency and 2 ports issue one each per cycle,
  so 4 × 2 = 8 independent accumulators is the minimum that keeps both ports
  busy, with no slack for any other stall.
- Loads: 9 load µops per 8 FMAs. A core with 2 load ports (Skylake, Coffee
  Lake) needs 4.5 cycles per k-step for the loads but 4 for the FMAs, so it can
  reach at most 8/9 of FMA peak. Lion Cove has 3 load ports and is not limited
  by this.
- Registers: 8 accumulators + 1 B vector + broadcasts fit in 16 YMM. The source
  loads the 8 C rows before the K loop; GCC 13 at `-O3` keeps the K loop free
  of stack accesses and spills 6 of those 8 C rows once, outside the loop
  (`objdump` of `inner_kernel_8x8`).
- Peak: 2 ports × 8 lanes × 2 flop = 32 flop/cycle for AVX2+FMA. BENCHMARKS.md
  reports the measured flop/cycle next to OpenBLAS on the same core.

**8×16 AVX-512** (`inner_kernel_8x16_avx512`). The same kernel with ZMM:
one 16-float B load, 8 broadcasts, 8 FMAs. On a core with two 512-bit FMA
units the peak is 64 flop/cycle, but 8 accumulators are still only
the latency minimum, and the 32 ZMM registers are mostly unused: production
SKX kernels keep 24–28 accumulators live (BLIS uses a 32×12 shape). Replacing
it is W3.

**6×16 AVX2 / 6×32 AVX-512** (`avx_matmul.cpp`, `simd_sgemm`). A second GEMM
with MC = 64, KC = 256, NC = 64, alpha folded into `pack_A`, and 12
accumulators (6 rows × 2 vectors). It is not reachable from Python; the C++
tests and the `bench` harness use it. It allocates its packing buffers inside
the `jc` loop.

### 2.5 Choosing the micro-kernel

The AVX2 and AVX-512 tables (§6) each have their own `sgemm`
(`sgemm_packed_avx2`, `sgemm_packed_avx512`); both use the AVX2 small-matrix
path. `sgemm(..., isa="scalar" | "avx2" | "avx512")` runs a given table for
one call and raises if this build or CPU cannot run it. The scalar table's
GEMM is a row-by-row loop with no blocking.

---

## 3. Elementwise kernels and normalization

All loops process 8 floats per iteration with unaligned loads (no alignment
requirement on NumPy inputs).

### 3.1 exp and tanh (`simd_math.hpp`)

`fast_exp_avx2(z)`: clamp z to ±88.376, `k = floor(z·log2(e) + 0.5)`,
`f = z − k·ln2` in two steps (Cody–Waite, ln2 split into a high part with
trailing zero bits and a low correction), degree-6 Horner polynomial for
`exp(f)` on |f| ≤ ln2/2 (Cephes coefficients), result `p(f)·2^k` built by
writing `k + 127` into the exponent field.

`tanh_avx2(y)`: clamp y to ±7.99, `e = exp(2y)`, `(e − 1)/(e + 1)` with the
division replaced by `rcpps` (12 bits) and one Newton–Raphson step
(`r·(2 − d·r)`, about 23 bits). There is no rational polynomial for tanh in the
code (an earlier version of this document described one).

### 3.2 GeLU (`intrinsic_gelu.cpp`)

`gelu(x, approximate=...)` has the same two definitions as
`torch.nn.functional.gelu`:

```
"none" (default)  0.5·x·(1 + erf(x/√2))
"tanh"            0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
```

`erf_avx2` is a rational minimax approximation `x·P(x²)/Q(x²)` (degrees 6 and
4 in x², XLA `ErfImpl32` coefficients) on x clamped to [−4, 4], with a true
`vdivps`: the quotient feeds `1 + erf`, where 1–2 ulp of error in erf near ±1
already dominates. For x < −10 both variants return −0 (|GeLU(−10)| =
7.6e-23); neither formula reaches −1 exactly in float32, so without this −inf
and very negative inputs would give large negative values. The last n mod 8
elements use masked load/store, so every element gets the same arithmetic.
The error bounds asserted by the tests are in the `gelu` docstring and
`docs/api/python_api.md`.

### 3.3 SiLU, ReLU

SiLU uses `σ(x) = 0.5·(1 + tanh(x/2))` with `tanh_avx2`, which needs no
`exp(−x)` and therefore cannot overflow; the scalar tail uses `1/(1 + exp(−x))`.
ReLU is `max(x, 0)`.

### 3.4 Softmax (`softmax_avx2.cpp`)

Three passes over a row: max (with NaN and +inf detection on the integer bit
patterns), `exp(x − max)` stored to the output with the sum accumulated in
double, then a multiply by `1/sum`. A row that contains NaN or +inf, or whose
entries are all −inf, returns NaN in every position, as `torch.softmax` does;
a partially −inf (masked) row is normal. The non-finite checks use bit
patterns because the file is compiled with `-ffast-math`, under which the
compiler may delete `isnan`/`isinf` tests (§5).

### 3.5 LayerNorm (`layer_norm_avx2.cpp`)

Three passes: mean (float → double, summed in double), variance of `x − mean`
(squared in float, summed in double), then `(x − mean)·inv_std` with optional
γ and β. Summing in double keeps the mean exact enough for inputs with a large
offset (tests use mean 50, std 100). LayerNorm is 0.82× PyTorch on the
reference laptop (BENCHMARKS.md); a single-pass (Welford) or fused two-pass
version is part of W3.

---

## 4. FP16- and BF16-input GEMM

`sgemm_f16` (F16C `vcvtph2ps`) and `sgemm_bf16` (BF16 → FP32 is exact:
zero-extend to 32 bits and shift left by 16) compute in FP32 and write FP32.
Both use the unpacked, unblocked structure of §2.1 for every size, converting
each B element every time it is used, with one accumulator per output strip
and no cache blocking. Their speed is not in BENCHMARKS.md yet (not measured
with the current protocol). Replacing them with conversion at pack time
followed by the FP32 micro-kernel (and
`vdpbf16ps` / VNNI paths where the hardware has them) is W4.

---

## 5. Compiler flags and floating point

Each instruction set has its own translation units, compiled with that ISA's
flags and nothing more (`CMakeLists.txt`, "ISA units"):

| Unit | Flags (GCC/Clang) | Sources |
|---|---|---|
| scalar | `-O3` (baseline x86-64: SSE2; or the target's default) | `dispatch/dispatch.cpp`, `kernels/scalar_kernels.cpp` |
| avx2 | `-mavx2 -mfma -mf16c -ffast-math -fno-finite-math-only` | AVX2 kernels, `kernels_avx2.cpp` |
| avx512 | `-mavx2 -mfma -mf16c -mavx512f -mavx512dq -mavx512bw -mavx512vl -ffast-math -fno-finite-math-only` | AVX-512 kernels, `kernels_avx512.cpp` |

The bindings and the C++ tests are compiled like the scalar unit, so the
extension imports and runs on any x86-64 CPU. `-DSIMD_ML_NATIVE=ON` adds
`-march=native` to every unit, for local benchmarking only; such a build runs
only on CPUs like the one it was built on.

Two checks keep the units honest:

- `ctest -R isa_units` (`tests/check_isa_units.py`) disassembles the object
  files: no VEX or EVEX instruction (first opcode byte `c4`/`c5`/`62`) and no
  LZCNT/TZCNT/POPCNT in the scalar unit, no EVEX in the AVX2 units, and no
  function emitted out of line (inline functions, template instantiations) by
  two units. The linker keeps one copy of such a function, so a scalar caller
  could otherwise end up in an AVX2 copy.
- Under Intel SDE the C++ tests and `selftest` run with an emulated CPU:
  `-nhm` (Nehalem, no AVX) uses only the scalar table, `-hsw` scalar and AVX2,
  `-skx` / `-spr` all three. This is how the AVX-512 kernels are tested on
  machines without AVX-512.

The SIMD units keep `-ffast-math` (reassociation, no errno, no signed-zero
or trapping guarantees) but add `-fno-finite-math-only`, so the compiler may
not assume that NaN and Inf never occur. That choice was measured: on the
reference laptop it was as fast as or faster than plain `-ffast-math` for every
elementwise kernel and faster than no fast-math for LayerNorm, with FP32 GEMM
unchanged (numbers in the PR that introduced it, #18; not a published
benchmark). Softmax still tests special values on bit patterns. Reassociation
means the compiler could give the vector body and the masked tail of an
elementwise kernel different rounding; with fixed flags the code no longer
depends on the build machine, and the tail-equals-body tests check it.

`tests/test_fp_env.py` checks that importing the module and running every
kernel on every usable table leaves FTZ and DAZ off in the host process
(`crtfastmath.o`, which sets them, is linked only into executables).

---

## 6. Runtime dispatch

`src/dispatch/kernels.hpp` defines one `KernelTable` per ISA: function
pointers for every kernel (FP32/FP16/BF16 GEMM, both GeLUs, ReLU, SiLU,
softmax row, LayerNorm row), plus the table's name, flags and GEMM tile. A
table may point at another table's functions: the AVX-512 table uses the AVX2
elementwise kernels, because only GEMM has an AVX-512 path.

`CpuFeatures::detect()` (`dispatch.cpp`) reads CPUID leaves 1 and 7 and XCR0
via XGETBV: YMM state needs XCR0 bits 1–2, ZMM state also bits 5–7, and every
feature flag that needs a register file includes that OS check. A table is
usable if its units were compiled in and the CPU has everything those units
were compiled for: AVX2 + FMA + F16C for avx2, plus AVX-512F/DQ/BW/VL for
avx512 (the compiler may emit BW/VL instructions in those units even where the
source does not use them).

The active table is chosen once, on first use: the widest usable one, or at
most the one named by the environment variable `SIMD_ML_ISA`
(`scalar`/`avx2`/`avx512`), which is how a narrower path is benchmarked or
tested on a wider CPU. `detected_isa()` and `build_info()["runtime_isa"]`
report it; `build_info()` also lists `isa_compiled`, `isa_usable` and each
usable table's flags. The private `_core._set_isa()` switches tables for the
whole process; `selftest` and the Python tests use it to check every usable
table. `detect_caches()` reads CPUID leaf 4 (Intel) or 0x8000001D (AMD) for
the core the calling thread runs on.

---

## 7. Python bindings

- **Zero copy.** Inputs must already be float32 (float16 / BF16 bit patterns
  for the low-precision GEMMs) and C-contiguous; nothing is converted
  silently. A caller-supplied `C` must be float32, C-contiguous, writeable and
  `[M, N]`; it is written in place (an earlier version cast it, wrote into a
  hidden copy and left the caller's array unchanged).
- **GIL.** Released for the duration of each kernel call.
- **Errors.** `DTypeError(TypeError, RuntimeError)` and
  `ArgumentError(ValueError, RuntimeError)`; the `RuntimeError` base keeps
  pre-0.3 `except RuntimeError` code working until 0.5.0.
- **Stubs.** `python/simd_ml_kernels/_core.pyi`; CI runs `mypy.stubtest` for
  names and `tests/test_stubs.py` for parameter names and defaults.

---

## 8. CI

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | push to main, pull requests | C++ build + ctest (GCC 12, Clang 15, GCC 13); Python 3.10–3.12: install, `selftest`, pytest (with PyTorch CPU), stubtest, examples, bench CLI smoke run, generated tables match JSON; ASan + UBSan (Clang 15) |
| `build-and-test.yml` | push to main, pull requests | GCC 12 with OpenBLAS-linked harnesses; OpenMP build; ASan + UBSan; macOS arm64 scalar build + ctest |
| `bench.yml` | weekly | `python -m simd_ml_kernels bench --json` on a shared runner, uploaded as an artifact; not used for any claim |
| `guardrail.yml` | push, pull requests, weekly | clang-tidy, pip-audit; report-only |

---

## Corrections to the documents this file replaces

For readers of older revisions: the merged-away files stated that GeLU's tanh
is a rational polynomial (it is exp-based, §3.1); that the B panel KC × NC is
8 KiB and L2-resident (it is 2 MiB; the 8 KiB block is the KC × NR micro-panel,
§2.3); that the 8×8 kernel's peak is 16 flop/cycle and its FMA chain 12 cycles
(32 flop/cycle and 4 cycles, §2.4); that the prefetch distance is 64 bytes (128,
§2.4); that `sgemm` went through a dispatch registry (until the per-ISA tables,
§6, the binding called the AVX2 GEMM directly); and that the C++ precision
tests ran kernels built without `-ffast-math` (they linked the fast-math
objects).

---

References: Goto & van de Geijn, *Anatomy of High-Performance Matrix
Multiplication*, ACM TOMS 34(3), 2008. Van Zee & van de Geijn, *BLIS*, ACM TOMS
41(3), 2015. Hendrycks & Gimpel, *GELUs*, arXiv:1606.08415. Ba et al., *Layer
Normalization*, arXiv:1607.06450. Intel 64 and IA-32 SDM vol. 2A (CPUID,
XGETBV).
