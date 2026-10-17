# GEMM Algorithm: Goto/BLIS 5-Loop Structure

simd-ml-kernels implements the Goto (2008) / BLIS-style packed GEMM in two complementary kernels. This document describes the shared algorithmic structure.

---

## The 5-Loop + 2 Packing Structure

The central insight from Goto & van de Geijn (2008) is that a high-performance GEMM must decomposes into **five nested loops** plus **two packing routines** that together ensure every level of the cache hierarchy is used efficiently:

```
for jc in 0..N step NC:          # Loop 1: N in NC tiles  (B panel → L3)
  for pc in 0..K step KC:        # Loop 2: K in KC tiles  (B_packed → L2)
    pack_b_panel(B[pc:, jc:])
    for ic in 0..M step MC:      # Loop 3: M in MC tiles  (A_packed → L2)
      pack_a_panel(A[ic:, pc:])
      for jr in 0..nc step NR:   # Loop 4: nc in NR steps (register tile)
        for ir in 0..mc step MR: # Loop 5: mc in MR steps (register tile)
          inner_kernel()          # ← hot loop: pure FMA
```

The two packing routines (`pack_b_panel`, `pack_a_panel`) convert the strided matrix access pattern into sequential streaming access, enabling the prefetcher and eliminating TLB misses in the inner kernel.

---

## Why Packing Matters

Without packing, the inner kernel accesses B with a stride of `ldb` floats between columns. For a 512×512 matrix (8 MB), this causes:
- Every B column access misses L1 (stride = 2 KB, L1 = 32–48 KB)
- TLB pressure from widely-scattered pages
- Observed throughput: ~5% of theoretical FMA peak

After packing B into a contiguous `kc × nr` panel:
- The inner kernel's B reads are sequential (unit stride)
- The hardware prefetcher can fully hide L2→L1 latency
- Observed throughput: 40–60% of theoretical FMA peak (single-threaded)

---

## Tile Size Derivation

| Constant | 8×8 AVX2 | 8×16 AVX-512 | 6×16 AVX2 | 6×32 AVX-512 | Derivation |
|---|---|---|---|---|---|
| MR | 8 | 8 (= MR512) | 6 | 6 | Register accumulators must fit in the register file |
| NR | 8 | 16 | 16 | 32 | float-width of the SIMD register (8 for YMM, 16 for ZMM) × accumulators-per-row |
| KC | 256 | 256 | 256 | 256 | Ac+Bc panels fit in L2 (256 KB): MC×KC×4 + KC×NC×4 ≤ L2 — unchanged by ISA |
| MC | 128 | 128 | 64 | 64 | Ac panel size; unchanged by ISA (MR is identical between AVX2/AVX-512 for both kernels) |
| NC | 2048 | 2048 | 64 | 64 | Large NC amortises B packing cost; unchanged by ISA |

Only NR changes between the AVX2 and AVX-512 variant of each kernel — MC/KC/NC
are cache-derived and independent of SIMD register width, so they carry over
unchanged. See ROADMAP.md §v0.8 for why the two kernels chose different
AVX-512 register strategies (single- vs dual-accumulator).

The NC=2048 choice in `avx2_gemm_packed.cpp` (vs NC=64 in `avx_matmul.cpp`) reflects a trade-off: larger NC means fewer B packing passes but larger L3 working set. For the Python-facing kernel, L3-resident B panels are acceptable since the outer loop can be parallelised with OpenMP (opt-in via `-DSIMD_ML_OPENMP=ON`; single-threaded by default).

---

## Panel Packing Layout

### B panel (`pack_b_panel`)

Input:  B[k×ldb] — strided row-major access  
Output: B_packed[(nc/NR) × kc × NR] — block-sequential

```
B_packed[block * kc * NR + p * NR + j] = B[p][block*NR + j]
```

Each `kc × NR` block is loaded sequentially by the inner kernel for one `jr` tile iteration. The NR-wide stride within each block enables a single `_mm256_loadu_ps` to load all 8 B values for one k-step.

### A panel (`pack_a_panel`)

Input:  A[m×lda] — strided row-major access  
Output: A_packed[mc × kc] — contiguous row-major

```
A_packed[i * kc + p] = A[i][p]
```

This layout allows scalar broadcast loads `_mm256_set1_ps(a_row[p])` for each row of the micro-kernel with sequential `p` access.

---

## Inner Kernel: 8×8 AVX2 (`avx2_gemm_packed.cpp`)

```
Registers:
  acc0..acc7   8 YMM   (8 row accumulators)
  c0..c7       8 YMM   (existing C values)
  b            1 YMM   (B panel column)

For each p in 0..kc-1:
  b = loadu(B_block + p*NR)            # 8 floats from B panel
  acc0 = fmadd(set1(a0[p]), b, acc0)   # row 0: scalar-broadcast A × vector B
  acc1 = fmadd(set1(a1[p]), b, acc1)   # row 1
  ...
  acc7 = fmadd(set1(a7[p]), b, acc7)   # row 7

C[0..7][0..7] += alpha * acc[0..7]     # write-back
```

8 FMAs per k-step, each computing one row of the 8×8 output tile. At 0.5 cycle/FMA throughput on two FMA ports, the theoretical ceiling is 16 FP32 ops/cycle = 56 GFLOPS at 3.5 GHz.

The 4× K-loop unrolling reduces branch overhead and exposes 32 FMAs to the out-of-order scheduler simultaneously, better hiding the 12-cycle FMA dependency chain.

---

## Inner Kernel: 6×16 AVX2 (`avx_matmul.cpp`)

An alternative tile shape that trades accumulator count for wider NR:

```
Registers:
  c0_lo, c0_hi ... c5_lo, c5_hi   12 YMM  (6 rows × 2 vectors each)
  b_lo, b_hi                        2 YMM  (16 floats from B panel)
  a_bcast                           1 YMM  (scalar broadcast)

For each k:
  b_lo = load(B + k*16),  b_hi = load(B + k*16 + 8)
  a_bcast = set1(A[k*6 + row])
  c_row_lo = fmadd(a_bcast, b_lo, c_row_lo)   # 8 cols
  c_row_hi = fmadd(a_bcast, b_hi, c_row_hi)   # 8 more cols
```

12 FMAs per k-step (6 rows × 2 vectors). With two FMA ports this processes 16 columns simultaneously — wider than the 8×8 kernel but with higher register pressure (15/16 YMM used).

The `alpha` scalar is folded into `pack_A` at packing time, eliminating a per-k multiply in the inner loop.

---

## Inner Kernel: 8×16 AVX-512 (`avx2_gemm_packed.cpp`)

Runtime-dispatched on AVX-512-capable hardware (falls back to the 8×8 AVX2
kernel above otherwise — see `gemm_packed_isa_is_avx512()`):

```
Registers:
  acc0..acc7   8 ZMM   (8 row accumulators, one per row)
  c0..c7       8 ZMM   (existing C values)
  b            1 ZMM   (B panel column, 16 floats)

For each p in 0..kc-1:
  b = loadu(B_block + p*NR512)          # 16 floats from B panel
  acc0 = fmadd(set1(a0[p]), b, acc0)    # row 0
  ...
  acc7 = fmadd(set1(a7[p]), b, acc7)    # row 7

C[0..7][0..15] += alpha * acc[0..7]     # write-back
```

Structurally identical to the 8×8 AVX2 kernel — same broadcast pattern, same
single-accumulator-per-row design — just twice the register width. One ZMM
(16 floats) already spans the full NR512=16 tile, so no dual-accumulator
split is needed (unlike the 6×32 kernel below). Register budget: 17/32 ZMM
used, 15 registers of headroom. See `docs/design/avx2_register_file.md
§AVX-512 Status` for the full register allocation rationale.

---

## Inner Kernel: 6×32 AVX-512 (`avx_matmul.cpp`)

Runtime-dispatched analogously via `avx_matmul_isa_is_avx512()`:

```
Registers:
  c0_lo,c0_hi .. c5_lo,c5_hi   12 ZMM  (6 rows × 2 halves each)
  b_lo, b_hi                    2 ZMM  (32 floats = 2 ZMM)
  a_broadcast                   1 ZMM  (scalar broadcast)

For each k:
  b_lo = load(B + k*32),  b_hi = load(B + k*32 + 16)
  a_bcast = set1(A[k*6 + row])
  c_row_lo = fmadd(a_bcast, b_lo, c_row_lo)   # cols 0-15
  c_row_hi = fmadd(a_bcast, b_hi, c_row_hi)   # cols 16-31
```

A 32-column tile exceeds one ZMM's 16-float width, so — unlike the 8×16
kernel above — each row genuinely needs two accumulators (lo/hi halves).
This dual-accumulator design is exactly what an earlier, disabled version
of this kernel got wrong: it declared a 32-wide tile but only ever computed
and wrote the "lo" half, silently leaving the "hi" half of every output
tile untouched after a `beta=0` memset. Both kernels above are covered by
permanent forced-full-coverage regression tests (`tests/test_gemm.cpp`,
`tests/test_gemm_packed.cpp`) specifically targeting this failure mode —
see ROADMAP.md §v0.8 for the full history.
