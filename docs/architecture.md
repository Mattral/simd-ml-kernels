# Architecture

simd-ml-kernels is organized as a layered C++17 library with a pybind11 Python front-end.
The diagram below shows the full dependency graph.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python / NumPy Layer                         │
│   simd_ml_kernels.*  (pybind11, GIL-release, shape validation)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
   ┌──────────▼──────────┐  │  ┌───────────▼────────────┐
   │  Activation Kernels  │  │  │  GEMM Kernels           │
   │  relu_avx2.cpp       │  │  │  avx2_gemm_packed.cpp  │
   │  silu_avx2.cpp       │  │  │  (8×8 primary kernel)  │
   │  softmax_avx2.cpp    │  │  │                         │
   │  layer_norm_avx2.cpp │  │  │  avx_matmul.cpp         │
   │  intrinsic_gelu.cpp  │  │  │  (6×16 alt. kernel)    │
   └──────────┬───────────┘  │  └───────────┬────────────┘
              │              │              │
   ┌──────────▼──────────────▼──────────────▼────────────┐
   │               Shared Math Infrastructure             │
   │   simd_math.hpp (fast_exp_avx2, tanh_avx2)          │
   │   cache_alloc.hpp (64-byte aligned allocator)       │
   │   cpuid.hpp / kernel_registry.hpp (ISA dispatch)    │
   └────────────────────────────────────────────────────┘
```

---

## Source layout

```
src/
├── kernels/
│   ├── simd_math.hpp            # Shared: Cody–Waite fast_exp + exp-based tanh
│   ├── cache_alloc.hpp          # 64-byte aligned allocator + STL adapter
│   ├── avx_matmul.{cpp,hpp}     # 6×16 AVX2/AVX-512 GEMM (reference kernel)
│   ├── intrinsic_gelu.{cpp,hpp} # GeLU: fast tanh-approx + erff oracle
│   └── activations/
│       ├── activations.hpp      # Public API for all activation functions
│       ├── relu_avx2.cpp        # ReLU: _mm256_max_ps (exact)
│       ├── silu_avx2.cpp        # SiLU: x·sigmoid(x) via fast tanh
│       ├── softmax_avx2.cpp     # Softmax: numerically stable, double accum
│       └── layer_norm_avx2.cpp  # LayerNorm: 3-pass AVX2, optional γ/β
│   └── gemm/
│       ├── avx2_gemm_packed.{cpp,hpp}  # 8×8 BLIS-style packed GEMM (primary)
│       ├── gemm_dispatcher.cpp         # CPUID-based runtime dispatch
│       └── naive_gemm.hpp              # Scalar reference (correctness oracle)
├── dispatch/
│   ├── cpuid.hpp                # x86 CPUID feature detection
│   └── kernel_registry.hpp     # Runtime kernel table (ISA → function ptr)
├── bindings/
│   └── pybind_entry.cpp         # pybind11 module: all Python-facing functions
├── main_bench.cpp               # RDTSC cycle-accurate benchmark
└── bench_stat.cpp               # Statistical benchmark (wall-clock, CI, JSON)
```

---

## Component responsibilities

### `simd_math.hpp` — shared transcendental math

The central shared header providing `fast_exp_avx2` and `tanh_avx2`.

`fast_exp_avx2` uses the Cody–Waite range-reduction algorithm:
1. `k = round(z · log₂e)` — integer, computed in the float exponent field
2. `f = z − k·ln2` (two-stage Cody–Waite to preserve accuracy)
3. `poly6(f)` — degree-6 minimax Horner polynomial for `exp(f)` over `[-ln2/2, ln2/2]`
4. `2^k · poly6(f)` — exact scaling via IEEE 754 exponent field manipulation

`tanh_avx2` uses the exp-based identity `tanh(y) = (exp(2y)−1)/(exp(2y)+1)` with a Newton–Raphson refined reciprocal instead of division.  Max absolute error: < 2×10⁻⁷.

Both `intrinsic_gelu.cpp` (GeLU) and `silu_avx2.cpp` (SiLU) include this header, eliminating duplicate polynomial code.

### GEMM kernels — two complementary implementations

**`avx2_gemm_packed.cpp`** (8×8 micro-kernel, used by Python dispatch):
- MR=8, NR=8: one YMM accumulator per row, one for B
- 5-loop BLIS structure: jc / pc / ic / jr / ir
- Packing buffers allocated once per call, reused across all k-panels
- 4× K-loop unrolling + software prefetch hints
- Optional OpenMP parallelism over the outermost jc-loop

**`avx_matmul.cpp`** (6×16 micro-kernel, educational reference):
- MR=6, NR=16: 12 YMM accumulators (6 rows × 2 vectors), maximises ILP
- Wider NR exposes more parallelism per k-step at the cost of register pressure
- alpha folded into `pack_A` at packing time (eliminates per-FMA multiply)
- 4× K-loop unrolling with software prefetch

Both implement the same Goto/BLIS algorithmic structure; see `docs/DESIGN.md §3`.

### Python bindings — `pybind_entry.cpp`

Key design decisions:
- **GIL release**: every kernel call releases the GIL (`py::gil_scoped_release`), allowing concurrent Python threads
- **Zero-copy**: `py::array_t<float, py::array::c_contiguous>` gives direct pointer access — no allocation or copy on the Python side
- **Validation layer**: shape, dtype, contiguity, and writeability are checked at the Python boundary before any kernel call, producing clear error messages
- **Output allocation**: `sgemm(A, B)` without `C` allocates a new NumPy array internally, matching `@` operator ergonomics

### Runtime ISA dispatch — `kernel_registry.hpp` / `gemm_dispatcher.cpp`

A lightweight function-pointer table populated once at startup via `std::call_once`:
1. `CpuFeatures::detect()` calls CPUID to detect AVX2, FMA, AVX-512, SSE4.2
2. The best available implementation is selected and stored in `KernelRegistry`
3. All subsequent calls go through the registry (one pointer dereference, no branch)

Current dispatch table:

| Detected ISA | GEMM kernel | GeLU kernel |
|---|---|---|
| AVX2 + FMA | `sgemm_packed` (8×8) | `gelu_avx2` |
| SSE4.2 only | `naive_sgemm` (scalar) | `nullptr` (scalar path in binding) |
| Scalar | `naive_sgemm` (scalar) | `nullptr` |
