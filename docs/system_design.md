# System Design

---

## Design Goals

| Goal | Status | Notes |
|---|---|---|
| Correct FP32 GEMM (Goto/BLIS style) | ✅ | Two implementations (8×8, 6×16) |
| Production-grade transcendental math | ✅ | Cody–Waite exp-based tanh, < 2e-7 error |
| Complete activation kernel set | ✅ | GeLU, ReLU, SiLU, Softmax, LayerNorm |
| Python NumPy integration | ✅ | Zero-copy, GIL-release, type stubs |
| Statistical benchmarking | ✅ | 95% CI, JSON output, regression gating |
| Runtime ISA dispatch | ✅ | AVX2/FMA auto-selected via CPUID |
| 64-byte aligned allocator | ✅ | Cache-line aligned, STL-compatible |
| Comprehensive test suite | ✅ | C++ + Python, 100+ test cases |

---

## Data Flow

### GEMM path

```
Python: simd_ml_kernels.sgemm(A, B, C, alpha, beta)
  │
  ▼ pybind_entry.cpp
Validate (shape / dtype / contiguity / writeability)
  │
  ▼ simd_ml::dispatch::sgemm()      [kernel_registry.hpp]
Runtime dispatch via KernelRegistry function pointer
  │
  ▼ simd_ml::gemm::sgemm_packed()   [avx2_gemm_packed.cpp]
scale_matrix_c(C, beta)             [one pass over C]
  │
  ├── Loop jc (N in NC=2048 tiles)
  │     ├── Loop pc (K in KC=256 tiles)
  │     │     pack_b_panel(B → B_packed)    [Bc panel, L2-resident]
  │     │     ├── Loop ic (M in MC=128 tiles)
  │     │     │     pack_a_panel(A → A_packed) [Ac panel, L2-resident]
  │     │     │     ├── Loop jr (nc in NR=8 steps)
  │     │     │     │     └── Loop ir (mc in MR=8 steps)
  │     │     │     │           inner_kernel_8x8()  ← hot path (AVX2 FMA)
  │     │     │     └──────────────────────────────
  │     │     └──────────────────────────────────────
  │     └──────────────────────────────────────────────
  └────────────────────────────────────────────────────
  │
  ▼
C updated in-place, returned to Python
```

### Activation path (GeLU example)

```
Python: simd_ml_kernels.gelu(x)
  │
  ▼ pybind_entry.cpp
Validate (dtype=float32, C-contiguous)
Allocate output array
  │
  ▼ gelu_forward_avx2()   [intrinsic_gelu.cpp]
Vectorized loop (8 floats/iteration):
  x3 = x * x * x
  inner = sqrt(2/pi) * (x + 0.044715 * x3)
  tanh_val = tanh_avx2(inner)     [via simd_math.hpp]
    └── e2y = fast_exp_avx2(2 * inner)   [Cody–Waite, 14 ops]
    └── tanh = (e2y - 1) / (e2y + 1)    [NR-refined reciprocal]
  output = 0.5 * x * (1 + tanh_val)
Scalar tail for n % 8 remaining elements
  │
  ▼
output ndarray returned to Python
```

---

## Memory Allocation Strategy

All SIMD kernels use 64-byte aligned buffers (`cache_alloc.hpp`):

- **Why 64 bytes?** One cache-line = 64 bytes = one AVX-512 register. Unaligned loads that straddle a cache-line boundary require two cache-line fetches.
- **C++17 aligned `operator new` / `delete`** (`std::align_val_t`): aligned heap allocation with no platform-specific branches
- **Packing buffers**: allocated once per `sgemm_packed` call, reused across all k-panel iterations (critical performance fix — earlier versions re-allocated inside the k-loop)
- **Python arrays**: `numpy.empty/zeros` allocates 64-byte aligned memory for large arrays. `is_aligned(arr)` verifies alignment. Unaligned inputs are handled by `_mm256_loadu_ps` (unaligned loads, same throughput on Haswell+)

---

## Runtime ISA Dispatch

Dispatch uses CPUID, executed once at startup via `std::call_once`:

```cpp
// gemm_dispatcher.cpp
const KernelRegistry& get_kernels() noexcept {
    std::call_once(g_registry_once, [] {
        CpuFeatures f = CpuFeatures::detect();   // CPUID leaf 1 + leaf 7
        if (f.has_avx2 && f.has_fma) {
            g_registry.sgemm = &gemm::sgemm_packed;
            g_registry.gelu  = &activations::gelu_avx2;
            g_registry.isa_label = "avx2";
        } else if (f.has_sse42) {
            g_registry.sgemm = &gemm_ref::naive_sgemm;
            // gelu falls back to scalar loop in pybind layer
        }
        // else: scalar fallback
    });
    return g_registry;
}
```

Overhead: one pointer dereference per dispatch call. Zero branches in the hot path.

---

## Threading Model

The library is **single-threaded by default**. OpenMP is an opt-in build flag:

```
# Without OpenMP (default):
sgemm_packed: serial for-loops

# With -DSIMD_ML_OPENMP=ON:
sgemm_packed: #pragma omp parallel for on the outermost jc-loop
              Each thread gets its own thread_local packing buffers
              (avoids false sharing on the packing writes)
```

The Python binding releases the GIL before calling any kernel, so Python-level threading (`threading.Thread`) will not cause stalls waiting for the kernel to finish.

---

## Benchmarking Infrastructure

Two complementary tools:

**`bench` (RDTSC cycle-accurate)**:
- Uses `LFENCE + RDTSC + RDTSCP + LFENCE` (Intel SDM Vol.2B §4.3)
- Reports `min(cycles)` over 10 reps: captures the best-case cache-warm execution
- Calibrates TSC frequency via `clock_gettime(CLOCK_MONOTONIC)` over 100 ms
- Outputs: cycles, time, GFLOPS, utilisation %

**`bench_stat` (statistical wall-clock)**:
- Uses `std::chrono::steady_clock` (nanosecond resolution)
- 5 warmup + 30 measurement reps by default
- Reports: min, mean, median, std-dev, 95% CI (t-distribution approximation)
- JSON output for longitudinal regression tracking
- CI checks that the benchmark tables match the committed JSON (`benchmarks/make_tables.py --check`)
