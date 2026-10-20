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
         AVX2 units    gemm/avx2_gemm_packed.cpp           FP32/FP16/BF16 GEMM: 6×16 kernel, direct and packed (§2, §4)
                       intrinsic_gelu.cpp, activations/    GeLU, ReLU, SiLU, softmax, LayerNorm
                       gemm/avxvnni_gemm.cpp               INT8 GEMM with AVX-VNNI (CPUID-gated, §4.2)
         AVX-512 units gemm/avx512_gemm_packed.cpp         FP32/FP16/BF16 GEMM, 12×32 kernel
                       gemm/avx512bf16_gemm.cpp            BF16 GEMM with vdpbf16ps (CPUID-gated, opt-in, §4.1)
                       gemm/avx512vnni_gemm.cpp            INT8 GEMM with AVX512_VNNI (CPUID-gated, §4.2)
           │
Shared   simd_math.hpp (fast_exp_avx2, tanh_avx2), cache_alloc.hpp (aligned allocation, prefetch)
```

Every Python entry point validates its arguments (`DTypeError` for dtype,
`ArgumentError` for shape, contiguity, writeability and values), allocates the
output if none was given, releases the GIL and calls one kernel through the
active table. Kernels never allocate Python objects; the only exception one
can throw is `std::bad_alloc` while growing the GEMM packing buffers (§2.2),
which pybind11 raises as `MemoryError`.

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
the kernel cannot read it in place: for `M ≤ 8` a third path computes dot
products along rows of A and B (the decode GEMV, §2.7), otherwise B is
packed. The dot path is well ahead of packing, and of OpenBLAS's own
`transb` GEMM, for a few rows (decode-style `x @ W.T`). The cut-off was
measured with both paths on the same core; the numbers are in `choose_path`.
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
(`-skx`, `-spr`) and on AVX-512 CI runners. Speed has been measured on one
AVX-512 machine, a 2-vCPU Emerald Rapids cloud VM whose clock and neighbours
are not controlled (BENCHMARKS.md,
`cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31`): 87–107% of OpenBLAS for
N = 128–2048, and below the 90% target at N = 2048 (87%). Bare-metal AVX-512
is not measured.

### 2.6 Choosing the table

The AVX2 and AVX-512 tables (§6) each have their own `sgemm`
(`sgemm_packed_avx2`, `sgemm_packed_avx512`); both send small products to the
AVX2 direct path. `sgemm(..., isa="scalar" | "avx2" | "avx512")` runs a given
table for one call and raises if this build or CPU cannot run it. The scalar
table's GEMM is a row-by-row loop with no blocking.

### 2.7 Decode GEMV: x·Wᵀ for one or a few rows

Generating one token multiplies one activation row x by every weight matrix
W of the model (`torch.nn.Linear` layout, `[N, K]`): `sgemm(x, W, trans_b=True)`,
`sgemm_f16(...)`, `sgemm_bf16(...)` with M = 1. Each weight is used once, so
the time is the time to read W from DRAM; arithmetic intensity is 2 flop per
weight (0.5 flop/byte in FP32, 1 in FP16/BF16), far below what a core can
compute while it waits. ("Dev runs" below are timings on the reference
laptop's P-core that decided a design choice; they are not in a JSON. The
published numbers are in BENCHMARKS.md.)

**The roof is read bandwidth.** `bench` measures two roofs on the pinned core,
over arrays of at least 4× the last-level cache (64 MiB minimum), and reports
every GEMV row as a percentage of both. The read stream is timed inside each
GEMV case, interleaved with its implementations, and that case's fastest
stream is its roof (`roof_read_gbs`); the triad is measured once per run. A
roof measured once per run drifted by several percent between runs on the
laptop, which moved every percentage with it. A case where any implementation
exceeds 105% of its own roof is marked `roof_unstable` and not published
(make_tables.py lists it under "Not shown", as for `--min-ghz`):

- *read stream*: the fastest read-only loop we have, including the GEMV
  access pattern, over one large float32 array (4 bytes per element, nothing
  written): its sum read as 2, 4, 8 and 16 parts side by side, and (AVX2
  kernels) as rows of 16 KiB with 4 or 8 rows walked together with 32-byte
  loads, the GEMV's own pattern, plus 8 rows with the next group prefetched.
  The roof is the fastest variant, which can only lower a kernel's
  percentage; it is a lower bound on the core's true read limit, not the
  limit itself. This is the primary roof for a GEMV, which only reads its
  big operand. With the part sums alone, the FP32 GEMV read 104–106% of the
  roof at 1×4096×11008 in three runs; in a dev run the 8-row pattern read 7%
  faster than the best part sum, and the FP32 GEMV 92–94% of it. In the
  publication run with every variant (`lnl-226v-wsl_2026-10-19_290b91baad26`) that case still had the FP32
  GEMV at 109.6% of its roof (31.8 GB/s, below the run's 35.3), so it is not
  shown; the Q8_0 case of the same shape was stable. Dev runs on the reference laptop: one stream reached 25–29
  GB/s, 2–8 streams 30–34 GB/s. Runs vary by several percent, so a kernel
  within about ±10% of 100% is at the roof.
- *STREAM triad*: `a = b + s·c`, 12 bytes per element (STREAM's convention).
  The store's read-for-ownership is real traffic that the convention does not
  count, so a read-only kernel can exceed 100% of the triad: our FP32 GEMV ran
  at 107–137% of it in dev runs, which is why it is not the GEMV's roof.

**One core does not saturate DRAM.** A core reaches its bandwidth limit
through the misses it can keep in flight (its fill buffers and prefetchers),
not through the memory controller's limit; all cores together read several
times more. So the single-threaded GEMV is judged against the single-core
roofs, and an all-core triad and read stream are recorded (OpenMP builds that
may use more than one CPU) for the threaded GEMV planned after the quantized
ones, which will be judged against those.

**Structure** (`dot_rows` / `gemm_dot_trans_b` in `avx2_gemm_packed.cpp` and
`avx512_gemm_packed.cpp`). W is never packed: packing copies all of W once per
call, which costs as much as using it (packed path at M = 1: 6–10 GB/s in dev
runs against 30–34 for the row-streaming path). Instead R rows of W are
walked together along K, R sequential streams for the hardware prefetchers;
each step reads 16 values per row (AVX2; 32 on AVX-512) into 2 accumulators
per row (one per row for R > 4 on AVX2, which has 16 registers). R = 4 for
FP32 and 8 for FP16/BF16, whose 32-byte row steps would otherwise keep half
as many bytes in flight (dev runs, 1×4096×4096 FP16: 28–31 GB/s with 4 rows,
31–34 with 8, level with FP32). FP16 and BF16 rows are converted in registers
(`vcvtph2ps`, or zero-extend and shift), so W is read at 2 bytes per element
and the result has the FP32 bound of §4.1. x is FP32 (FP16/BF16 x is converted
once per call, K values); each R-row block of W is reused from L1/L2 for every
row of x. K tails use masked loads (AVX2 FP32, AVX-512) or a copied tail
(AVX2 16-bit), so nothing past a row is read. Software prefetch 256–1024 bytes
ahead on every row changed nothing beyond run-to-run noise, so there is none.

**Cut-off.** With more rows of x, every row re-reads the current R-row block
from cache with 2 loads per FMA, and packing B wins from about 9 rows at
K = 4096 (FP32) to 12–14 (FP16/BF16); at K = 1024 from 9 (BF16) to 14 (FP32).
`kDotRowsMax` = 8 wins or ties in every measured case; the numbers are in
`choose_path`. A kernel that reuses each W vector for several rows of x would
move that cut-off; batched decode is not a target of this work yet.

The AVX-512 table runs the same structure with ZMM registers (row counts
chosen on the AVX2 core). Its speed is not measured: the only AVX-512 machine
available is the cloud VM, which moved between hosts and whose single-core
memory bandwidth varied from 5.8 to 11 GB/s from run to run (dev runs, not
in a JSON), so a GEMV percentage of a roof measured once per run would not
mean anything there; the per-case roof is meant to make such a run usable (not
run there yet). The scalar table computes the same dot products, one
accumulator per element.

**Measured** (BENCHMARKS.md, `lnl-226v-wsl_2026-10-18_ecc219baa1f2`, one
P-core; roofs measured once per run, before the per-case roof: 31.8 GB/s read
stream, 29.6 GB/s triad), at 1×4096×4096,
1×4096×11008 and 1×2048×8192:

- FP32: 31.1–31.4 GB/s, 97–99% of the read stream; 1.03–1.05× PyTorch's
  `F.linear` and 1.00–1.20× OpenBLAS's sgemv. All three read at about the
  roof, so FP32 has little left to gain on one core.
- FP16 and BF16: 28.7–31.4 GB/s, 90–99% of the read stream; 1.78–1.92×
  PyTorch's `F.linear` on the same 16-bit weights, which reads at 15.5–17.5
  GB/s (49–55% of the read stream). Half the bytes of FP32 at about the same
  GB/s, so close to half the time of our FP32 GEMV.

### 2.8 Q8_0 decode GEMV (`gemv_q8_0`)

`quantize_q8_0(W)` stores W [N, K] in ggml's Q8_0 format, byte for byte
`block_q8_0`: each run of 32 weights is a 34-byte block, a binary16 scale
d = max|w|/127 then 32 int8 quants q = roundf(w/d), as
`quantize_row_q8_0_ref` (tested against ggml's own function when
`SIMD_ML_GGML_LIB` or `SIMD_ML_GGML_TEST_BACKEND_OPS` points at a llama.cpp
build, and against a NumPy port always). The quantization runs in the scalar
unit, without `-ffast-math`, so `amax / 127` and `w · (1/d)` round exactly as
the C reference does. One case differs: where 1/d overflows (max|w| below
127/FLT_MAX), ggml converts ±inf to int8, which C leaves undefined; we store
q = 0 (the binary16 scale is 0 there anyway). The bytes are 34/32 = 1.0625
per weight, 27% of FP32 and 53% of FP16/BF16.

**x is quantized once per call** to the same blocks (its quants equal
`quantize_q8_0(x)`'s), but its scale dx = max|x|/127 stays FP32: rounding it
to binary16, as ggml does, would add error and can underflow for small
activations or overflow above 8.3·10⁶, for no saving (K values against N·K
weights). The quantization is in the scalar unit (exact IEEE arithmetic, see
above); for K = 11008 it is under 0.1% of the bytes the GEMV reads.

**Dot product per block.** y[n] = Σ_b fp16(dw)·dx·Σ_i qw·qx: the 32-term
integer sum is exact (|Σ| ≤ 32·127² < 2²⁴, so also exact as FP32), then one
multiply by the scale product and one FMA into an 8-lane FP32 accumulator
per row. Both operands are signed, but `vpdpbusd` and `vpmaddubsw` multiply
unsigned by signed bytes. Two ways out: ggml's sign trick, |x|·sign(w, x) =
x·w per byte (`vpabsb`, `vpsignb`), or the offset trick `sgemm_u8s8` uses
for its zero point, (x + 128)·w = x·w + 128·w, which needs Σw per block (one
more `vpdpbusd` against ones per row and block). We use the sign trick: |x|
is computed once per x block and shared by all rows, and `vpsignb` is one
instruction per 32 weights, the same count as the offset's extra dot but
without a per-block correction. It is exact because Q8_0 never stores −128
(|w/d| ≤ 127), so |x| fits a byte and `vpsignb` never negates −128, and on
AVX2 without VNNI `vpmaddubsw`'s int16 pair sums stay at most 2·127² = 32258
(no saturation). Instructions per block and row: one unaligned 32-byte load
(34-byte stride: about half the loads split a cache line), `vpsignb`, the dot
(`vpdpbusd`; `vpmaddubsw` + `vpmaddwd` on AVX2), `vcvtdq2ps`, the scale
(`vpbroadcastw` + `vcvtph2ps`, one `vmulps` by dx) and one FMA: 8–10 µops for
34 bytes. At the laptop's read roof (about 32 GB/s at 4.3 GHz, 7.5 bytes per
cycle) one block arrives every 4.5 cycles, so the loop needs about 2 µops per
cycle of an 8-wide core: memory, not instructions, sets the speed.

**Structure** (`q8_0_rows.hpp`, one copy per unit with that unit's dot): 8
rows walked together along K, as the FP32 GEMV (§2.7), with x's block
(x, |x|, dx) loaded once per 8 rows. Dev runs (AVX-VNNI, % of the read roof
measured next to them, no software prefetch): 1 row 50–63%, 2 rows 55–63%,
4 rows 60–76%, 8 rows 70–91%. One row at a time is ggml's structure
(`ggml_vec_dot_q8_0_q8_0` per output) and runs at about ggml's speed (below).
Unlike the FP32 GEMV, software prefetch helps here: one `prefetcht0` per two
blocks for the same row of the *next* 8-row group took the three shapes from
67–91% to 90–100% of the roof in dev runs (8192×2048, rows of 2176 bytes,
gained most: 67–72% to 90–93%); prefetching 512 or 1024 bytes ahead in the
same row changed nothing. A row is only 34–68 cache lines long here, so a new
group's 8 streams otherwise start cold every few microseconds.

Paths (`build_info()["q8_0_path"]`): `avx_vnni` (VEX `vpdpbusd`), `avx2`
(`vpmaddubsw` + `vpmaddwd`), `avx512_vnni` (EVEX `vpdpbusd` on YMM: a ZMM
register would need two loads and a lane insert per two blocks plus a
per-lane scale vector, in a loop bound by memory; correct under SDE, speed
not measured), `scalar` (exact int sums per block). All single-threaded.

**Error.** Against the exact sum of the quantized operands (every path,
tested): |y − y64| ≤ 2·(K/32 + 4)·u·Σ_b |dw·dx|·Σ_i |qw·qx|. Against the FP32
product x·Wᵀ, the quantization of both operands dominates; per weight
|w − ŵ| ≤ d·(0.5 + 2⁻¹⁶ + 127·2⁻¹¹) = 0.5621·d (rounding w/d, then d to a
normal binary16) and |x − x̂| ≤ 0.5001·dx, so |y − x·Wᵀ| ≤
Σ_b [0.5622·dw·Σ|x| + 0.5001·dx·Σ|ŵ|] plus the kernel bound (tested on
standard-normal data). Measured on the benchmark's standard-normal W and x
(`err_vs_fp32` in `lnl-226v-wsl_2026-10-19_290b91baad26`): relative L2 error of y 0.75–0.77% at the three
shapes, largest error 0.66–0.75% of max|y|; with the weights quantized and x
in FP32 (the FP32 GEMV on the dequantized weights), 0.53–0.54%: about 70% of
the error is the weights', the rest x's.

**ggml baseline.** llama.cpp's `test-backend-ops perf` runs one MUL_MAT
(q8_0 × f32, m = N, n = 1, k = K) given as a `--test-file` line, built
CPU-only Release with default flags (`GGML_NATIVE`, OpenMP, llamafile and
repack on; test-backend-ops does not use the repacked weight buffers). It
has no thread option and asks for hardware_concurrency()/2 threads, so it
runs with `OMP_THREAD_LIMIT=1` under `taskset` on our core. Thread check, in
the last 1.0 s before it exits (inside its ≥ 1 s timed loop): at most one
thread, and threads other than the main one at most 5% of its CPU time
(process minus main-thread ticks from /proc, which also counts threads that
lived between samples). Its `threads_max` is 5 and its other threads use
7–26% of the main thread's CPU over the whole run: 4 test-data threads per
tensor (`init_tensor_uniform`), all before timing; in the timed window, one
thread. It times its own way (the op repeated on the same tensors for ≥ 1 s,
mean per run; we take the median of three such runs), so it is a separate
column, not interleaved with ours.

**Measured** (BENCHMARKS.md, `lnl-226v-wsl_2026-10-19_290b91baad26`, one P-core, path `avx_vnni`), at
1×4096×4096, 1×4096×11008 and 1×2048×8192:

- 31.7, 32.4 and 29.4 GB/s: 92–95% of each case's read roof; 3.49–3.61× faster
  than our FP32 GEMV and 1.77–2.05× than our BF16 GEMV in the same case (they
  read 3.8× and 1.9× the bytes); 3.41–3.70× PyTorch's FP32 `F.linear` on the
  dequantized weights (not a quantized kernel).
- ggml (llama.cpp v0.7.0, one thread): 887–2750 µs, 17.4–20.1 GB/s, 52–63% of
  the same roofs; ours is 1.46–1.82× faster (ggml timed its own way, above).
  ggml's `MUL_MAT` with n = 1 runs llamafile's `tinyBLAS_Q0_AVX::gemm<4, 1>`
  (`ggml/src/ggml-cpu/llamafile/sgemm.cpp`): 4 weight rows per pass, two
  `vpsignb` per row and block (|w| and sign(x, w); we compute |x| once per
  block for all rows), two binary16 scale conversions per row and block (one
  here), and no software prefetch. The bytes are the same (34 per 32 weights;
  x quantized per call in both), so the gap is memory parallelism: in dev runs
  our loop with 4 rows and no prefetch read 60–76% of the roof, close to
  ggml's 52–63%, 8 rows 70–91%, and the next-group prefetch 90–100%.

### 2.9 Q4_0 decode GEMV (`gemv_q4_0`)

`quantize_q4_0(W)` stores W [N, K] in ggml's Q4_0 format, byte for byte
`block_q4_0`: each run of 32 weights is an 18-byte block, a binary16 scale d
then 16 bytes of 4-bit quants, byte j holding value j in its low nibble and
value 16 + j in its high nibble; value = d·(q − 8). As
`quantize_row_q4_0_ref`: max = the first value of largest magnitude (sign
kept), d = max/−8, q = min(15, trunc(w·(1/d) + 8.5)). The value of largest
magnitude therefore maps to q = 0 (−8·d = max), and values on the other side
with |w/d| ≥ 7.5 clamp to q = 15 (+7·d). A zero block stores d = −0.0 and
q = 8. Tested byte for byte against blocks written by llama.cpp v0.7.0's own
quantization (`ggml_quantize_chunk`, committed fixture
`tests/data/q4_0_ggml_v0.7.0.npz`, 20 rows of 256 values covering ties, zero
blocks, binary16-subnormal and near-overflow scales, exact halves), against
ggml's library when `SIMD_ML_GGML_LIB` is set, and against a NumPy port. The
quantization runs in the scalar unit without `-ffast-math` and with
`-ffp-contract=off`: w·id and + 8.5 are two roundings in ggml's reference
(baseline x86-64, no FMA), and a fused multiply-add truncates differently
next to an integer. Where 1/d overflows (max|w| below 8/FLT_MAX) ggml's C
code is undefined; we store q = 8 (the scale is binary16 zero there anyway).
The bytes are 18/32 = 0.5625 per weight: 53% of Q8_0, 14% of FP32.

**x** is quantized once per call to Q8 blocks with FP32 scales, the same
code as `gemv_q8_0` (§2.8); ggml's `vec_dot_q4_0_q8_0` also takes Q8_0
activations.

**Why the Q8_0 loop is not enough.** At the laptop's read roof one Q4_0
block of one row arrives every ~0.55 ns (18 bytes at ~32 GB/s), and the
Q8_0 loop's structure costs ~3 cycles per row and block (9 vector µops;
measured with the weights in L2 at 3.2 GHz: 0.98 ns). The first Q4_0 loop,
one row per register like Q8_0 (`vbroadcasti128` + `vpsrlvd` + `vpand` for
the nibbles, then as Q8_0), ran at 55–64% of the read roof in dev runs:
bound by instructions, not memory. Q8_0 hid the same cost behind 34 bytes.

**Two rows per register** (`q4_0_rows.hpp`). Rows r and r + 1 share one YMM
register, lanes 0–3 and 4–7, so the conversion, the scale and the FMA serve
two rows. Per row pair and block:

| step | instructions | per row and block |
|---|---|---:|
| the two rows' 16 quant bytes | `vmovdqu xmm` + `vinserti128` (load) | 1 (+1 load) |
| nibbles: low, high | `vpand`; `vpsrlw` + `vpand` | 1.5 |
| Σ q·x (low half with x[0..15], high with x[16..31]) | 2 `vpdpbusd` (AVX2: 2 × `vpmaddubsw` + `vpmaddwd` + `vpaddd`) | 1 (3) |
| two binary16 scales, 4 copies each | `vmovd` + `vpinsrd` (loads) + `vpshufb` | 1 |
| 8 FP32 scales × dx | `vcvtph2ps` (2 µops), `vmulps` | 1.5 |
| accumulate | `vcvtdq2ps`, `vfmadd231ps` | 1 |
| | | **6.5** (Q8_0 loop: 8, ggml's Q4_0: ~12) |

Plus, once per block for all 8 rows: x's two 16-byte halves broadcast to
both lanes, and the offset −8·Σx (two dots of 8 with x, negated). Σ (q − 8)·x
= Σ q·x − 8·Σ x: q in [0, 15] is the dot's unsigned operand and x's quants
the signed one, so no sign trick is needed (ggml subtracts 8 from q and then
uses `vpsignb` twice, as for Q8_0), and −8·Σx is the starting accumulator of
the first `vpdpbusd`, free with VNNI. `vpmaddubsw`'s int16 pair sums are at
most 2·15·127 = 3810 (no saturation). An odd last row uses the one-row loop.

**Unpacking nibbles** costs 1.5 instructions per row and block (3 per pair:
`vpand`, `vpsrlw`, `vpand`). Dev measurement, weights in L2 (512×2048 and
11008×4096, AVX-VNNI, alternating builds): with the three instructions
removed (wrong results, timing only), 1.8–2.3 cycles per row and block instead
of 2.3–2.6, about 0.1–0.35 cycles (5–15%) of the loop; the two-rows layout
took the loop itself from ~3.1 to ~2.4 cycles. With the weights in DRAM the
two-rows loop read 65–85% of the read roof in dev runs (one row per
register: 55–64%).

**Rows and prefetch, re-measured for Q4_0** (dev runs, AVX-VNNI, % of a read
roof timed next to them, two runs each): rows per pass 4, 8, 12 and 16 were
within run-to-run noise of each other (with the next-group prefetch every 4
blocks: 65–81% at 4, 67–81% at 8, 63–80% at 12, 60–77% at 16), so 8 stays
(4 accumulators; 16 rows would not fit the 16 YMM registers on AVX2). The
next-group prefetch matters more than for Q8_0: without it 48–66% for every
row count. Spacing (8 rows): one `prefetcht0` per row every 1, 2, 3, 4 or 8
blocks gave 64–85%, 68–84%, 69–83%, 62–81% and 51–73%; every 2 blocks
(36 bytes, so every cache line of the next group's rows is touched) is used.
Bytes per block dropped from 34 to 18, so the prefetch every 2 blocks of
Q8_0 (68 bytes) became one every 2 blocks here at about twice the rate per
byte; every 8 blocks (144 bytes) skips lines and loses ~10 points.

Paths: the same as `gemv_q8_0` (`build_info()["q8_0_path"]` names it):
`avx_vnni`, `avx2` (`vpmaddubsw`), `avx512_vnni` (EVEX `vpdpbusd` on YMM;
correct under SDE, speed not measured), `scalar`. Single-threaded.

**Error.** Against the exact sum of the quantized operands (every path,
tested): |y − y64| ≤ 2·(K/32 + 4)·u·Σ_b |dw·dx|·Σ_i |(qw − 8)·qx|. Against
x·Wᵀ: per weight |w − ŵ| ≤ d·(1 + 8·2⁻¹¹) (rounding w/d, 0.5·d, or the clamp
at +7, up to d; then d to a normal binary16) and |x − x̂| ≤ 0.5001·dx, so
|y − x·Wᵀ| ≤ Σ_b [1.0045·|dw|·Σ|x| + 0.5001·dx·Σ|ŵ|] plus the kernel bound
(tested on standard-normal data). Measured on the benchmark's standard-normal
W and x (`err_vs_fp32` in `lnl-226v-wsl_2026-10-20_42f64c7d76b3`): relative L2 error of y 8.37–8.76% at the
three shapes, largest error 7.59–8.92% of max|y|; with the weights quantized
and x in FP32, 8.36–8.75%: the 4-bit weights are almost all of it (Q8_0:
0.75–0.77%, §2.8).

**ggml baseline**: as §2.8 (`test-backend-ops perf`, MUL_MAT q4_0 × f32,
one thread, the same thread check). ggml's n = 1 Q4_0 path is again
`tinyBLAS_Q0_AVX::gemm<4, 1>`: 4 rows per pass, no prefetch, and per row and
block a lane-crossing denibble (`vpsrlw` on 128 bits + `vinsertf128` +
`vpand`), `vpsubb` 8, two `vpsignb`, the dot, `vcvtdq2ps`, two scalar
binary16 conversions, a multiply and a broadcast, and the FMA: about 12
instructions where we spend 6.5. Note: Q4_0 weights at 1×4096×4096 and
1×2048×8192 are 9.4 MB, close to the 8 MiB L3, and test-backend-ops reuses
the same weights on every run (ours rotate copies), so part of ggml's reads
there may hit the cache; the comparison favours ggml on those two shapes.

**Measured** (BENCHMARKS.md, `lnl-226v-wsl_2026-10-20_42f64c7d76b3`, one P-core, path `avx_vnni`). Only
1×4096×11008 passed the clock check in that run (1×4096×4096 and 1×2048×8192
each had a clock sample below 3.6 GHz in all three attempts; their rows stay
in the JSON, not in the tables; three publication runs on this laptop that
day all had such dips). At 1×4096×11008: 1028 µs, 24.7 GB/s, 79% of the
case's read roof; 1.67× faster than our Q8_0 GEMV and 6.30× than our FP32
GEMV in the same case (they read 1.9× and 7.1× the bytes); 5.79× PyTorch's
FP32 `F.linear` on the dequantized weights. ggml (llama.cpp v0.7.0, one
thread): 2221 µs, 37% of the roof; ours is 2.16× faster. The remaining gap to
the roof is the loop's instruction cost (above): Q4_0 is about balanced
between memory and compute on this core, where Q8_0 is memory-bound.

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

SiLU is `x / (1 + exp(−x))` with `fast_exp_avx2` and a correctly rounded
`vdivps`; for −x > 88 the denominator is set to +inf, so the result is −0 and
SiLU(−inf) is NaN, as in PyTorch. An earlier version used
`x·0.5·(1 + tanh(x/2))`: `tanh_avx2` clamps at ±7.99, so its error grew
without bound below x = −16 (1e-4 at x = −1000), and `1 + tanh` cancels for
x ≪ 0. The exp form has a relative bound, `|err| ≤ 1e-35 + 1e-6·|SiLU(x)|`
for every finite x (tested; an exhaustive sweep measured 2.9e-7 relative), and
was faster in the dev timing at the top of `silu_avx2.cpp`.
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

Two passes for the statistics and one to normalize: pass 1 sums `x − x[0]`,
pass 2 sums `(x − μ)²`, pass 3 writes `(x − μ)·inv_std` with optional γ and β.
Each statistics pass keeps four independent 8-float accumulators and adds
them into a double every 1024 elements, so float rounding grows with 32 terms
per lane rather than with the row length, and shifting by `x[0]` keeps the
summed values near the spread of the row rather than its offset.

This replaced a kernel that summed in double through one dependent add chain
per pass (latency-bound). A single-pass Welford variant (per-lane updates,
Chan merge) was measured against it and against two-pass: it was no faster
than the old kernel for rows up to 1024 elements (its per-element update costs
more than reading an L1-resident row twice) and less accurate on long rows
(float lane state). The comment at the top of the file has the numbers;
BENCHMARKS.md has the published comparison with PyTorch.

Accuracy: `|err| ≤ 1e-5 + 1e-4·|ref|` for rows with `|mean| ≤ 30·std`
(tests: up to 65536 elements, and a first element far from the mean). Beyond
that, rounding the mean to float32 dominates (about `6e-8·|mean|/std`), as it
does in PyTorch.

---

## 4. Low-precision GEMM

### 4.1 FP16- and BF16-input GEMM

`sgemm_f16` and `sgemm_bf16` take 2-byte inputs, accumulate in FP32 and write
FP32. Both conversions to FP32 are exact (F16C `vcvtph2ps`; BF16 is the top
half of an FP32, so zero-extend and shift left by 16), so the result is the
FP32 GEMM of the converted inputs, with the same error bound.

**Conversion at pack time.** The AVX2 and AVX-512 units run the FP32 paths of
§2 with the conversion moved into packing: `pack_a` / `pack_b` and the loops
are templates over the input element type, and each element is converted
once, when it is copied into its micro-panel (8 values per `vcvtph2ps` or
per zero-extend + shift, 16 on AVX-512). The micro-kernels are the FP32 ones,
unchanged; the FP32 instantiation is the same code as before. The direct path
(the FP32 shapes of §2.1) converts each 6-row strip of A into a small buffer
and B either in registers (one strip: M ≤ 6, so a single row reads B once at
2 bytes per element) or once into a buffer (more strips).

The previous kernels converted every B element on every use, with one
accumulator per output strip and no cache blocking: 12–17% of the FP32 GEMM
at N = 256–1024 on the reference laptop (dev measurement, not in a JSON).
Measured now (BENCHMARKS.md, `lnl-226v-wsl_2026-10-18_045918e51839`):

- Square N = 128–2048: FP16 at 95–100% and BF16 at 99–101% of our FP32 GEMM;
  FP16 1.41–1.46× PyTorch's float16 `matmul`; BF16 0.97–1.16× PyTorch's
  bfloat16 `matmul` (it is slightly ahead at N = 1024 and 2048).
- On the Emerald Rapids cloud VM (AVX-512 table,
  `cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31`): FP16 at 91–101% and BF16
  at 99–104% of our FP32 GEMM; FP16 0.89–1.25× PyTorch's float16 `matmul`
  (it is ahead from N = 512 up); BF16 0.14–0.72× PyTorch's bfloat16 `matmul`
  (see the `vdpbf16ps` note below for why).
- 1×4096×4096 with B stored [K, N] (bound by reading B): FP16 at 180% and
  BF16 at 162% of our FP32 GEMM, because B is half the bytes; both lose to
  PyTorch (0.63× and 0.58×; 0.62× for FP16 again in
  `lnl-226v-wsl_2026-10-18_ecc219baa1f2`). That path walks B in 16-column
  strips down all of K. With the weights stored [N, K] (`trans_b=True`, the
  `torch.nn.Linear` layout a model uses) the decode GEMV of §2.7 reads at
  90–99% of the read stream, 1.78–1.92× PyTorch's `F.linear`.

Memory: A and B are stored at 2 bytes per element instead of 4. The packing
buffers are FP32 and the same size as for the FP32 GEMM (per thread, at most
(MC+6)·KC + KC·(NC+16) floats, §2.2), so the saving is in the operands, not in
the working set. On compute-bound shapes there is no speed gain on this
hardware (AVX2 has no 16-bit FMA): FP16/BF16 here are memory savings at about
the FP32 speed, and faster only where reading B dominates.

Cost on narrow products: with N ≤ 16 and large M·K the FP32 kernel issues one
A broadcast per 2 FMAs, so converting A first is a large share of the work
(dev timing: 70–82% of the FP32 GEMM at 2048×16×2048 and 4096×8×4096; not in
the JSON).

**`vdpbf16ps` (AVX512_BF16), opt-in.** On CPUs with AVX512_BF16 (CPUID leaf
7.1 EAX bit 5: Cooper Lake, Sapphire Rapids and later, Zen 4 and later) and
with `SIMD_ML_BF16_DOT=1` in the environment (read once), the AVX-512
table's `sgemm_bf16` runs packed single-threaded products through
`avx512bf16_gemm.cpp`: the 12×32 tile with K taken in pairs, A broadcast as
one 32-bit pair per row, B packed as 32 columns × 2 BF16 per k-pair (two ZMM
vectors). Each `vdpbf16ps` adds two exact BF16 products per FP32 lane, each
add rounded to nearest even, so the error bound is unchanged. The instruction
ignores MXCSR: BF16 subnormal inputs (|x| < 2⁻¹²⁶) are read as zero and
subnormal results flushed, unlike the conversion path; the tests expect that
on this path. Small products keep the AVX2 direct path, and more than one
OpenMP thread uses the threaded conversion path. Correct under SDE `-spr`.
Measured on an Emerald Rapids cloud VM (`cloud-xeon-emr-2vcpu_2026-10-18_9029f1566ddb`):
this path runs at 64–73% of our FP32 GEMM on square N = 128–2048 (the
conversion path is at 90–101% of FP32 for FP16 in the same run), so on that
CPU it is slower than the path it replaces. It is therefore off by default,
which also keeps BF16 results identical on every CPU. Not diagnosed, and not a
priority: in the same run PyTorch's BF16 `matmul` reached about 250
flop/cycle, above the AVX-512 ceiling of 128 BF16 flop/cycle (two `vdpbf16ps`
per cycle × 16 lanes × 2 products; 64 for FP32 FMA). That rate implies AMX
tiles (presumably through oneDNN; not checked), which this library does not
use, so the BF16 work that pays here is the memory-bound single-row case (W5).

### 4.2 INT8 GEMM (`sgemm_u8s8`)

`C[i, j] = a_scale · b_scales[j] · Σ_k (A[i, k] − zp) · op(B)[k, j]` with A
uint8 (activations: one scale and zero point per tensor), B int8 (weights:
symmetric, one scale per output column), float32 out: PyTorch's quantized
linear without the requantization of the output. `trans_b` takes B as
`torch.nn.Linear` stores it ([N, K]).

**VNNI.** `vpdpbusd` multiplies 4 consecutive uint8 of one operand by 4 int8
of the other and adds the 4 products to an int32 lane: 32 multiply-adds per
YMM instruction (64 per ZMM), against 8 (16) for `vfmadd231ps`, on the same
two ports. The kernels keep the FP32 tiles (6×16 with 12 YMM accumulators for
AVX-VNNI, `avxvnni_gemm.cpp`; 12×32 with 24 ZMM accumulators for
AVX512_VNNI, `avx512vnni_gemm.cpp`) with K taken in quads. A micro-panels
hold one 32-bit word (4 k values) per row and quad, broadcast to every lane;
B micro-panels hold one word per column and quad (4 B rows interleaved with
`vpunpck{l,h}bw/wd`, or 8×8 word transposes for `trans_b`), followed by the
panel's column sums, computed with `vpdpbusd` against all-ones bytes. The
zero point is subtracted at the end of each K block: Σ (a − zp)·b = Σ a·b −
zp·Σ b. Within a block the int32 sums are exact (|Σ| ≤ KC·255·128 < 2³¹ for
KC ≤ 65793); KC is four times the FP32 KC (one byte per element, so the
KC×16 B micro-panel takes the same half of L1D), at most 4096. Each block's
sum is converted to FP32, scaled by `a_scale·b_scales[j]` and added to C, so
the only roundings are per block.

**Without VNNI** (Haswell to Coffee Lake, Skylake-SP): `vpmaddubsw` would
saturate its int16 pair sums (2·255·127 > 32767), so the AVX2 table
converts instead: A − zp and B become FP32 while packing (exact), the FP32
kernel computes the sums, and the columns are scaled at the end. That path
is a memory saving (1-byte operands), not a speed-up.

**Scalar table:** exact int64 sums. Error bound on every path:
`|C − C_exact| ≤ 2·(K+2)·u·|a_scale·b_scales[j]|·Σ_k |A − zp|·|op(B)|`, tested
against an exact int64 reference (not float64 BLAS: under SDE `-skx`,
OpenBLAS's float64 matmul of these integers was wrong).

`build_info()["u8s8_path"]` says which path runs: `avx512_vnni` (AVX-512
table and AVX512_VNNI), `avx_vnni`, `fp32_convert` or `scalar`. All paths are
single-threaded.

**One row with B [N, K]** (M = 1, `trans_b`: x·Wᵀ for one token) skips the
packing on every x86 path (`int8_rows.hpp`): the rows of B are read in place,
4 at a time (8 on AVX512_VNNI, which has 32 registers for the two int32
accumulators per row), with the next 4-row group's same row prefetched as in
the Q8_0 GEMV (§2.8). Σ a·b and Σ b come from two `vpdpbusd` per 32 bytes
(A is uint8 and B int8: the instruction's own operand order, no sign trick);
without VNNI, a − zp and b are widened to int16 and multiplied with
`vpmaddwd` (exact). Lanes are int32 and K is summed in chunks of 2¹⁸ (one
step adds at most 130560 to a lane), each chunk's lanes into int64, so the
sum is exact; the bound above holds. Results: below.

Measured (BENCHMARKS.md, `lnl-226v-wsl_2026-10-18_19aeaf83cd36`; `avx_vnni`,
x·Wᵀ with W stored [N, K]):

- Square N = 128–2048: 235–416 GOP/s, 2.39–3.68× our FP32 GEMM (3.50–3.68× from
  N = 512 up) and 1.66–2.04× PyTorch's `quantized.linear` (x86 engine,
  prepacked weights). 98.0 and 97.5 operations per cycle (2 per
  multiply-add) at N = 1024 and 2048; the core's `vpdpbusd` throughput, and
  so the fraction of peak, is not measured.
- The FP32 conversion path (what a CPU without VNNI runs) at 0.95–1.03× our
  FP32 GEMM: memory savings at FP32 speed, no gain in time.
- 1×4096×4096 (one row times a weight matrix), before the row path: 1.15× our
  FP32 (its dot-product path, §2.1) but **0.46× PyTorch**; the conversion
  path **0.47× our FP32**. Every call packed all of B (and, for the
  conversion path, wrote it as FP32), which costs as much as using it once.
  With the row path (`lnl-226v-wsl_2026-10-19_290b91baad26`): 58.4 GOP/s, 4.09× our FP32 GEMV and
  **1.36× PyTorch** (`avx_vnni`); the int16 path of AVX2 CPUs without VNNI,
  65.5 GOP/s (4.59× our FP32). Both read B at about the read roof, so the
  VNNI dot buys nothing at M = 1.
- `avx512_vnni` on the Emerald Rapids cloud VM
  (`cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31`): 1.55–2.83× our FP32 GEMM
  at N = 128–2048, but 0.67–0.75× PyTorch's `quantized.linear` from N = 512
  up (1.26–2.46× below that), and 0.13× PyTorch for 1×4096×4096. The 12×32
  VNNI tile has not been tuned on that CPU.

---

## 5. Compiler flags and floating point

Each instruction set has its own translation units, compiled with that ISA's
flags and nothing more (`CMakeLists.txt`, "ISA units"):

| Unit | Flags (GCC/Clang) | Sources |
|---|---|---|
| scalar | `-O3` (baseline x86-64: SSE2; or the target's default) | `dispatch/dispatch.cpp`, `kernels/scalar_kernels.cpp` |
| avx2 | `-mavx2 -mfma -mf16c -ffast-math -fno-finite-math-only` | AVX2 kernels, `kernels_avx2.cpp` |
| avx512 | `-mavx2 -mfma -mf16c -mavx512f -mavx512dq -mavx512bw -mavx512vl -ffast-math -fno-finite-math-only` | AVX-512 kernels, `kernels_avx512.cpp` |
| avx512bf16 | the avx512 flags and `-mavx512bf16` (built if the compiler accepts it) | `gemm/avx512bf16_gemm.cpp`; no table of its own: the AVX-512 table calls it after a CPUID check and only with `SIMD_ML_BF16_DOT=1` (§4.1) |
| avxvnni | the avx2 flags and `-mavxvnni` (built if the compiler accepts it) | `gemm/avxvnni_gemm.cpp`; called by the AVX2 table's `sgemm_u8s8` after a CPUID check (§4.2) |
| avx512vnni | the avx512 flags and `-mavx512vnni` (built if the compiler accepts it) | `gemm/avx512vnni_gemm.cpp`; called by the AVX-512 table's `sgemm_u8s8` after a CPUID check (§4.2) |

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
  `-skx` / `-spr` all three (`-spr` also has AVX512_BF16, so it runs the
  `vdpbf16ps` BF16 path). This is how the AVX-512 kernels are tested on
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
pointers for every kernel (FP32/FP16/BF16 GEMM, INT8 GEMM, both GeLUs, ReLU,
SiLU, softmax row, LayerNorm row), plus the table's name, flags, GEMM tile and
block sizes. A
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
| `ci.yml` | push to main, pull requests | C++ build + ctest (GCC 12, Clang 15, GCC 13); Python 3.10–3.12: install, `selftest`, default table matches the runner's CPU flags, pytest (with PyTorch CPU), stubtest, examples, bench CLI smoke run, generated tables match JSON; ASan + UBSan (Clang 15) |
| `build-and-test.yml` | push to main, pull requests | GCC 12 with OpenBLAS-linked harnesses; OpenMP build; ASan + UBSan; macOS arm64 scalar build + ctest |
| `wheels.yml` | pull requests touching the build, manual, called by `publish.yml` | cibuildwheel: manylinux_2_28 x86_64 and macOS arm64 wheels, CPython 3.9–3.13, and an sdist; each wheel installed in a clean environment and `selftest`ed. On pull requests, artifacts only |
| `publish.yml` | manual (owner), from `main` | builds through `wheels.yml`, then uploads to TestPyPI or PyPI with Trusted Publishing (no API token). 0.3.0rc1 (TestPyPI) and 0.3.0 were published this way |
| `bench.yml` | weekly, manual | `python -m simd_ml_kernels bench --json` on a shared runner, uploaded as an artifact; not used for any claim |
| `guardrail.yml` | push, pull requests, weekly | clang-tidy (fails on errors), pip-audit and codespell (report-only), license file check |

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
