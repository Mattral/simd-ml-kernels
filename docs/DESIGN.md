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
         AVX2 units    gemm/avx2_gemm_packed.cpp           FP32 GEMM: 6×16 kernel, direct and packed
                       gemm/{f16,bf16}_gemm.cpp            FP16- / BF16-input GEMM (unpacked, see §4)
                       intrinsic_gelu.cpp, activations/    GeLU, ReLU, SiLU, softmax, LayerNorm
         AVX-512 units gemm/avx512_gemm_packed.cpp         FP32 GEMM, 12×32 kernel
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

### 2.1 One micro-kernel, two ways to feed it

The AVX2 table has one micro-kernel, 6×16 (`kernel_6x16` in
`avx2_gemm_packed.cpp`): 6 rows of C × two YMM vectors = 12 accumulators. It
reads `A(r, p)` at `A[r·a_rs + p·a_cs]` and row `p` of B at `B + p·b_rs`, so the
same code runs

- **direct**, on A and B in place (`a_rs = lda`, `a_cs = 1`, `b_rs = ldb`), and
- **packed**, on the micro-panels of §2.2 (`a_rs = 1`, `a_cs = 6`, `b_rs = 16`;
  this instantiation has the strides as compile-time constants, which the
  packed path needs to reach its speed; the comment at the kernel has the
  measurement).

`sgemm_avx2_prefers_direct(M, N, K)` picks direct for products up to 80³, for
`N ≤ 16` (one panel of columns) and for `M = 1` (each element of B is used
once, so packing it costs as much as using it). With `trans_a` the direct
path only swaps A's strides. With `trans_b` B has no contiguous rows of N, so
the kernel cannot read it in place: for `M ≤ 16` a third path computes dot
products along rows of A and B (4 rows of B at a time, 8 accumulators, masked
K tail), otherwise B is packed. The dot path is well ahead of packing, and of
OpenBLAS's own `transb` GEMM, for a few rows (decode-style `x @ W.T`); the
measurement is in `choose_path`. The cut-off was measured with
both paths on the same kernel; the numbers are in the function's comment.
Row and column remainders use the same kernel with fewer rows (a template
parameter, 1-6) and masked B loads and C accesses (`n < 16`), so nothing
outside C's `M × N` window is read or written.

Before this design (until 0.2.0) small products (`max(M, N, K) ≤ 128`) ran a
loop with one accumulator per 8 columns, a chain of dependent FMAs bound by
FMA latency, and every packed call allocated and freed its 2.1 MiB packing
buffers.

### 2.2 Goto/BLIS 5 loops

```
for jc in 0..N step NC:
  for pc in 0..K step KC:
    pack_b(B[pc:pc+KC, jc:jc+NC])        → 16-column micro-panels, KC×16 each
    for ic in 0..M step MC:              (OpenMP: loops 3-4 in chunks, §2.4)
      pack_a(A[ic:ic+MC, pc:pc+KC])      → 6-row micro-panels, k-major
      for jr in 0..NC step 16:
        for ir in 0..MC step 6:
          kernel_6x16: C[ir:ir+6, jr:jr+16] += alpha · Ap · Bp
```

```
Bp[(j / 16)·16·kc + p·16 + (j % 16)] = op(B)(p, j)   (zero-padded to 16 columns)
Ap[(i / 6)·6·kc  + p·6  + (i % 6)]   = op(A)(i, p)   (zero-padded to 6 rows)
```

`op(A)(i, p)` is `A[i][p]`, or `A[p][i]` with `trans_a` (A stored K×M);
`op(B)(p, j)` is `B[p][j]`, or `B[j][p]` with `trans_b` (B stored N×K). The
packing routines read either layout, so a transpose never costs a copy:
with `trans_a` the 6 values of one k-step are adjacent; with `trans_b` each
panel column is a row of B, transposed 8×8 in registers (a scalar gather
measured 59–86% of OpenBLAS for M = 16..96, the register transpose 94–98%).

`beta` is applied to C once before the loops (`beta = 0` writes zeros and never
reads C, so NaN in an uninitialised C does not propagate). The packing
buffers are per thread (`thread_local`), grown on demand and kept, so a call
allocates nothing once a thread has run a GEMM of that size; with the Lunar
Lake blocking they reach at most (MC+6)·KC + KC·(NC+16) floats, 4.8 MiB.

### 2.3 Block sizes from the caches

`dispatch::gemm_blocking(MR, NR)` derives the block sizes once, from CPUID
leaf 4 (Intel) or 0x8000001D (AMD) of the core that runs the first GEMM:

| Block | Rule | Lunar Lake P-core (48 KiB L1D, 2.5 MiB L2, 8 MiB L3) | 32 KiB L1D, 256 KiB L2 |
|---|---|---|---|
| KC | KC×16 B micro-panel fills ½ of L1D | 384 | 256 |
| MC | MC×KC packed A fills ¼ of L2 | 426 | 60 |
| NC | KC×NC packed B fills ½ of the last-level cache | 2720 | depends on L3 |

The B micro-panel is reused for every 6-row A micro-panel of the MC block, so
it has to stay in L1 next to the streaming A micro-panel and the C tile. The
packed A block is reused across the whole jr loop from L2. MC barely matters
once it is past a few dozen rows (B is re-read from L3 once per MC rows, 2/MC
bytes per flop); half of L2 was measurably slower than a quarter. The comment
at the top of `avx2_gemm_packed.cpp` has the sweep. `build_info()["sgemm_blocking"]`
and the `build` block of every benchmark JSON record the values used. On a
hybrid CPU the first GEMM's core decides (pin benchmarks).

The AVX-512 table uses the same loops and rules with its 12×32 kernel:
KC = 128 with a 32 KiB L1D (Skylake-SP), 192 with 48 KiB (Ice Lake, Sapphire
Rapids).

### 2.4 Threads (OpenMP builds)

OpenMP is a build option (`-DSIMD_ML_OPENMP=ON`), off by default; whether to
turn it on by default waits for a scaling table on a non-hybrid 8-core CPU.
In an OpenMP build one packed GEMM call uses up to `get_num_threads()`
threads: `OMP_NUM_THREADS` (else one per logical CPU) until
`set_num_threads(n)` is called. The small-product paths (direct, dot) stay
single-threaded.

All threads of a call work on the same NC-wide column block, one block after
the other. For each KC panel they pack the KC×NC B panel together (each a
contiguous run of micro-panels; one shared copy, sized for the last-level
cache) and meet at a barrier. The block is then cut into chunks
(`gemm_chunks`): about 4 per thread, along M (mc rows, a multiple of MR, at
most MC), and also along N (at least 128 columns) when M has fewer than 2
MR-row micro-panels per thread. Chunks are handed out with
`schedule(dynamic, 1)`: on a hybrid CPU a P-core takes more chunks than an
E-core instead of waiting for it at the next barrier. Each thread packs the
rows of A of its chunk into its own buffer; the implicit barrier at the end
of the chunk loop keeps the B panel until every chunk is done.

- **Result independent of the thread count.** Each element of C is computed by
  one thread with the same kernel and the same order over K as with one
  thread, so 1, 2, 3, 4 and 8 threads give bitwise-equal results (tested).
- **Small products stay on fewer threads.** `gemm_threads` gives each thread
  at least 2 MFLOP (about 20 µs at 100 GFLOP/s); below that, waking threads
  and the two barriers per KC panel cost more than they save.
- **No nested teams.** A call made inside an enclosing OpenMP parallel region
  runs on one thread; each calling thread has its own buffers, so concurrent
  calls are safe (tested).
- **Buffers.** The calling thread's buffers are grown before the parallel
  region (an exception cannot cross it): one B panel and one A block per
  thread, each rounded to 64 bytes. `set_num_threads` does not call
  `omp_set_num_threads`, so the host program's own OpenMP regions are not
  affected.

The constants (2 MFLOP per thread, 4 chunks per thread) were chosen from dev
runs on the hybrid Lunar Lake laptop under WSL2 (the comments in
`dispatch.cpp` give them); scaling is not published until it is measured on
the i9-9900 workstation.

### 2.5 Why 6×16

- **Latency × ports:** 2 FMAs per cycle with 4-cycle latency (5 on Haswell)
  need at least 8 independent accumulators. 6×16 has 12, so a late load does
  not stall both FMA pipes; the previous 8×8 kernel had exactly 8.
- **Registers:** 12 accumulators + 2 B vectors + 1 broadcast = 15 of 16 YMM.
- **Loads per FMA:** 2 B loads + 6 broadcasts per 12 FMAs (0.67), against
  9 per 8 (1.13) for 8×8. On a core with 2 load ports (Skylake, Coffee Lake)
  8×8 could reach at most 8/9 of FMA peak.
- **Measured:** the three candidate shapes (8×8, 6×16, 4×24) in one packed
  driver on the Lunar Lake P-core; 6×16 was best or tied from N = 128 up. The
  comment at the top of `avx2_gemm_packed.cpp` has the numbers;
  docs/BENCHMARKS.md has the published comparison with OpenBLAS.
- Peak: 2 ports × 8 lanes × 2 flop = 32 flop/cycle for AVX2+FMA; BENCHMARKS.md
  reports the measured flop/cycle next to OpenBLAS on the same core.

**AVX-512: 12×32, 24 ZMM accumulators** (`avx512_gemm_packed.cpp`). Two
512-bit FMA units with 4-cycle latency need 8 accumulators; the previous 8×16
kernel had exactly 8 and left 23 of 32 ZMM registers unused. 12 rows × 2
vectors = 24 accumulators + 2 B vectors = 26 registers; per k-step 2 B loads
and 12 broadcasts feed 24 FMAs (0.58 loads per FMA, 7 cycles of loads on
2 load ports against 12 of FMAs). It is the row-major view of BLIS's
Skylake-X SGEMM shape (32×12 column-major). 14×32 (28 accumulators) also fits;
the row count is one constant. Correctness is checked under Intel SDE
(`-skx`, `-spr`) and on AVX-512 CI runners; its speed is measured only on real
AVX-512 hardware (BENCHMARKS.md, "not measured yet" until then).

### 2.6 Choosing the table

The AVX2 and AVX-512 tables (§6) each have their own `sgemm`
(`sgemm_packed_avx2`, `sgemm_packed_avx512`); both send small products to the
AVX2 direct path. `sgemm(..., isa="scalar" | "avx2" | "avx512")` runs a given
table for one call and raises if this build or CPU cannot run it. The scalar
table's GEMM is a row-by-row loop with no blocking.

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
  machines without AVX-512. It is a local check before a PR, not a CI job:
  Intel's download server refuses (HTTP 403) the scripted downloads CI would
  need, and the SDE license does not allow keeping a copy in this repository.
  In CI, runners with AVX-512 (they vary between AMD Zen 3, Zen 5 and Intel
  Xeon) run the AVX-512 table natively, and a test fails if they do not.

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
| `wheels.yml` | pull requests touching the build, manual | cibuildwheel: manylinux_2_28 x86_64 and macOS arm64 wheels, CPython 3.9–3.13, each installed in a clean environment and `selftest`ed. Artifacts only, never published |
| `bench.yml` | weekly | `python -m simd_ml_kernels bench --json` on a shared runner, uploaded as an artifact; not used for any claim |
| `guardrail.yml` | push, pull requests, weekly | clang-tidy, pip-audit; report-only |

---

## Corrections to the documents this file replaces

For readers of older revisions: the merged-away files stated that GeLU's tanh
is a rational polynomial (it is exp-based, §3.1); that the B panel KC × NC is
8 KiB and L2-resident (it is 2 MiB; the 8 KiB block is the KC × NR micro-panel,
§2.3); that the 8×8 kernel's peak is 16 flop/cycle and its FMA chain 12 cycles
(32 flop/cycle and 4 cycles, §2.5); that the prefetch distance is 64 bytes (128,
§2.5); that `sgemm` went through a dispatch registry (until the per-ISA tables,
§6, the binding called the AVX2 GEMM directly); and that the C++ precision
tests ran kernels built without `-ffast-math` (they linked the fast-math
objects).

---

References: Goto & van de Geijn, *Anatomy of High-Performance Matrix
Multiplication*, ACM TOMS 34(3), 2008. Van Zee & van de Geijn, *BLIS*, ACM TOMS
41(3), 2015. Hendrycks & Gimpel, *GELUs*, arXiv:1606.08415. Ba et al., *Layer
Normalization*, arXiv:1607.06450. Intel 64 and IA-32 SDM vol. 2A (CPUID,
XGETBV).
