# Cache Hierarchy and Tile Size Derivation

simd-ml-kernels's GEMM implementation explicitly targets all three cache levels.
The tile constants MC, KC, NC are derived to keep specific panels resident at
specific cache levels during the 5-loop computation.

---

## Cache Targets by Loop Level

| Loop | Panel | Fits in | Size (bytes) |
|---|---|---|---|
| Loop 2 (pc): K-tiles | Bc panel (KC × NC) | L2 | KC × NC × 4 = 256 × 8 × 4 = 8 KB |
| Loop 3 (ic): M-tiles | Ac panel (MC × KC) | L2 | MC × KC × 4 = 128 × 256 × 4 = 128 KB |
| Loops 4+5: register tiles | MR × NR accumulators | L1 / registers | 8 × 8 × 4 = 256 bytes |
| Loop 1 (jc): N-tiles | Bc stream | L3 | NC × K × 4 = 2048 × K × 4 |

The large NC=2048 in `avx2_gemm_packed.cpp` is intentional: B panels larger
than L2 are acceptable as long as they fit in L3, where streaming prefetch
latency (~40 cycles) is tolerable since the inner micro-kernel produces 8 FMAs/step
(≈ 4 cycles at full throughput) before the next B panel line is needed.

---

## Derivation: KC and MC for L2

For a dual-resident L2 target (both Ac and Bc fit together):

```
Ac panel size = MC × KC × 4 bytes
Bc panel size = KC × NC_small × 4 bytes  (NC_small = NR × n_blocks)

Target: Ac + Bc ≤ L2_size
  128 × 256 × 4 + 256 × 8 × 4 = 131072 + 8192 = 139264 bytes ≈ 136 KB

Typical L2 = 256 KB (Skylake) → 136 KB leaves comfortable headroom for
instruction cache, OS data, and speculative prefetch staging.
```

On Zen 4 (L2 = 1 MB) or Alder Lake (L2 = 1.25 MB per P-core), these tile
sizes are deeply L2-resident. A larger MC or KC could be used profitably on
those microarchitectures — this is listed in ROADMAP.md as future auto-tuning work.

---

## Alignment and Cache-Line Crossing

All packing buffers use 64-byte alignment (`cache_alloc.hpp`):

- 64 bytes = 1 cache line = 1 AVX-512 register
- `_mm256_load_ps` (aligned) has the same throughput as `_mm256_loadu_ps`
  (unaligned) on Haswell+, but unaligned access *can* cause extra load-port
  traffic when the load straddles a cache-line boundary
- The packing routines in `avx2_gemm_packed.cpp` guarantee that packed panel
  rows are cache-line aligned, so all inner-kernel B loads are guaranteed
  single-cache-line accesses

---

## Software Prefetching

The inner kernel issues `prefetch_l1` hints 4 iterations ahead in the K-loop:

```cpp
prefetch_l1(b_ptr + 4 * NR);   // pull next B panel chunk into L1
```

**Why 4 iterations?**
- Skylake L2→L1 fill latency ≈ 12 cycles
- Inner kernel processes 8 FMAs/iteration × 0.5 cy/FMA = 4 cycles/iteration
- 12 cycles ÷ 4 cycles/iter = 3 iterations minimum → 4 gives a safety margin

The prefetch hint is a *hint*, not a guarantee. On CPUs with a strong hardware
prefetcher (Ice Lake, Zen 3+), the software hint is usually redundant but harmless.
On older or lower-power cores it can provide a measurable throughput improvement.
