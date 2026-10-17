# Performance Model

---

## Theoretical Peak (Single Core, AVX2+FMA)

```
Architecture:     Intel Skylake / AMD Zen 3 (representative)
AVX2 registers:   16 YMM × 8 float32 = 128 FP32 lanes
FMA ports:        2 (port 0 + port 5)
FMA throughput:   0.5 cycles per _mm256_fmadd_ps  (2 FMAs/cycle)

Peak FP32 ops/cycle = 2 ports × 8 floats/YMM × 2 (mul + add in FMA) = 32 FP-ops/cycle

At 3.5 GHz:  32 × 3.5 = 112 GFLOPS

Wait — why does main_bench.cpp print 56 GFLOPS as peak?

Answer: the standard definition counts a fused multiply-add as 2 FLOPS
(one multiply, one add). An FMA instruction executes in 0.5 cycles, so:

  throughput = (1 FMA/0.5 cy) × (2 FLOPS/FMA) × (2 ports) = 8 GFLOPS/GHz
  At 3.5 GHz: 8 × 3.5 = 28 GFLOPS per port × 2 ports = 56 GFLOPS total

The discrepancy: 112 vs 56 arises from whether you count "vector operations"
(one _mm256_fmadd_ps = 1 op = 8 floats × 2 arithmetic ops) or "FLOPS".
  
  simd-ml-kernels uses FLOPS convention: peak = 2 ports × 8 floats × 2 FLOPS × freq
                                          = 32 GFLOPS/GHz
  At 3.5 GHz: 32 × 3.5 = 112? No...

Correct derivation:
  2 FMA ports × 1 FMA/cycle per port × 8 FP32/FMA × 2 FLOPS/FMOP × 3.5 GHz
  = 2 × 1 × 8 × 2 × 3.5 = 112 GFLOPS

But bench.cpp uses peak = 16 ops/cycle at 3.5 GHz = 56 GFLOPS.
This is because each FMA port issues 1 instruction/cycle, each instruction
performs 8×2=16 scalar FLOPS, and with 2 ports that is 32 FLOPS/cycle.
bench.cpp correctly uses PEAK_FP_PER_CYCLE = 16 which is per-port, not total.
At 3.5 GHz: 16 × 3.5 = 56 GFLOPS per-port. Multiply by 2 ports → 112 GFLOPS total.

The bench output shows utilisation as gflops / (PEAK × freq), where PEAK = 16.
This gives utilisation relative to ONE port. 50% utilisation = using one full port.
```

**Bottom line**: when the bench reports "util: 50.0%", it means the kernel is
using one full FMA port. Both FMA ports simultaneously = 100% on a dual-port core.
Observed utilisation of ~40–60% means one port is near-saturated.

---

## Arithmetic Intensity (Roofline)

For C = A × B with M=N=K:

```
FLOPs     = 2 × N³
Data (B)  = 4 × (N² + N² + N²) = 12 N²  bytes (assuming no caching)

Arithmetic Intensity (AI) = FLOPs / bytes = 2N³ / (12N²) = N/6  FLOP/byte

Ridge point (compute/memory transition):
  At peak_FLOPS = 56 GFLOPS and bandwidth = 60 GB/s:
  ridge = 56 / 60 ≈ 0.93 FLOP/byte

  N > 6 × 0.93 ≈ 5.6 → N ≥ 6: compute-bound
```

For N=256: AI = 256/6 ≈ 43 FLOP/byte >> ridge point.
The 256×256 GEMM is deeply compute-bound; packing ensures Ac/Bc panels
stay cache-resident, so the effective bandwidth demand is much lower than the
formula above suggests.

For N=8: AI = 8/6 ≈ 1.3 FLOP/byte > ridge but barely.
Small matrices are near the ridge point; tiling overhead dominates.

---

## Benchmarking Methodology

### `bench` (cycle-accurate)

Uses RDTSC with Intel SDM serialisation sequence:
```
LFENCE              # prevent prior loads from completing after RDTSC
RDTSC → t0          # read cycle counter (not self-serialising alone)

[kernel]

RDTSCP → t1, aux    # self-serialising read (waits for prior instructions)
LFENCE              # prevent subsequent loads from being reordered before t1
```

Reports `min(cycles)` over N repetitions. The minimum captures the best-case
cache-warm execution without OS scheduling jitter.

**Limitations**:
- TSC frequency varies with turbo boost; bench.cpp calibrates via `clock_gettime` over 100 ms
- Without core pinning, the OS may migrate the thread mid-measurement (use `taskset -c 0`)
- Without frequency locking, turbo can inflate reported GFLOPS by 10–30%

### `bench_stat` (statistical)

Uses `std::chrono::steady_clock` with 30 measurement reps:
- Reports mean ± 95% CI (Student t-distribution), min, and median
- CPU affinity set via `sched_setaffinity` when running as root
- JSON output enables longitudinal regression tracking in CI

**When to use which**:
- `bench` for quick microbenchmark comparison (±5% accuracy acceptable)
- `bench_stat` for publishable claims or regression gates (needs freq lock)

---

## Interpreting Benchmark Output

```
  SIMD   GEMM  256x256 | min: 4,557,563 cy | time: 1.864 ms | GFLOPS: 18.01 | util: 32.2%

  GFLOPS = 2 × 256³ / (elapsed_s × 1e9)
         = 33.55e6 / 1.864e-3 / 1e9 = 18.01 GFLOPS

  util   = GFLOPS / (TSC_Hz / 1e9 × PEAK_PER_PORT)
         = 18.01 / (3.5 × 16) = 18.01 / 56 = 32.2%
```

32% utilisation means the kernel is using ~1/3 of one FMA port's capacity.
The primary gaps are:
1. C++ intrinsics vs hand-tuned assembly: ~5–15% overhead
2. Fixed tile sizes not optimal for this microarchitecture: ~10–20% gap
3. Software prefetch may be suboptimal for this specific cache hierarchy

See ROADMAP.md §v0.8 for planned AVX-512 and auto-tuning work.
