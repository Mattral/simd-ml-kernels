# simd-ml-kernels

**Transparent Hand-Vectorized SIMD Microkernels for Machine Learning**

[![CI](https://github.com/Mattral/simd-ml-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/simd-ml-kernels/actions/workflows/ci.yml)
[![Weekly Benchmark](https://github.com/Mattral/simd-ml-kernels/actions/workflows/bench.yml/badge.svg)](https://github.com/Mattral/simd-ml-kernels/actions/workflows/bench.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python ≥3.9](https://img.shields.io/badge/Python-≥3.9-3776ab)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/Preprint-Zenodo-orange)](https://zenodo.org/records/20616502)

simd-ml-kernels is an open-source C++17 library of clean, well-documented
hand-vectorized microkernels for core ML primitives. It deliberately occupies
the middle ground between toy textbook code and opaque production BLAS:
production techniques (panel packing, register blocking, software prefetching)
but every design decision explained in source comments and documentation.

**Preprint**: [IntrinsicML: Transparent Hand-Vectorized SIMD Microkernels for ML Workloads](https://zenodo.org/records/20616502)

**Naming**: one name everywhere: repository and PyPI distribution `simd-ml-kernels`,
import `simd_ml_kernels`. The project was previously called IntrinsicML (the preprint
title), the repository `SIMD-Microkernels-for-ML-Workloads` (GitHub redirects old links),
and the import `simd_kernels`, which still works but emits a `DeprecationWarning` and
will be removed in 0.5.0. See [CHANGELOG.md](CHANGELOG.md).

---

## Kernels

| Kernel | Implementation | Notes |
|--------|---------------|-------|
| **GEMM** (SGEMM) | `avx_matmul.cpp`, `gemm/avx2_gemm_packed.cpp` | Goto/BLIS 5-loop, panel packing, MR×NR register blocking, AVX2/AVX-512 dispatch |
| **GeLU** | `intrinsic_gelu.cpp` | Exact (erf, rational approximation) by default or tanh formula (`approximate="tanh"`), as in PyTorch; max abs error ≤ 2e-6 |
| **ReLU** | `activations/relu_avx2.cpp` | `_mm256_max_ps`, exact |
| **SiLU** | `activations/silu_avx2.cpp` | `x * sigmoid(x)` via fast tanh, 8 floats/cycle |
| **Softmax** | `activations/softmax_avx2.cpp` | Numerically stable (max-subtraction) |
| **LayerNorm** | `activations/layer_norm_avx2.cpp` | 3-pass, double accumulation, optional γ/β |

All kernels have AVX2 fast paths with scalar fallbacks for non-AVX2 systems.

---

## Quick Start

### Python (recommended)

```bash
pip install .
python -m simd_ml_kernels selftest   # every kernel vs a float64 reference; exits 1 on failure
python -m simd_ml_kernels info       # compiled ISA, flags, commit, CPU features
```

```python
import numpy as np
import simd_ml_kernels as sk

A = np.random.randn(256, 256).astype(np.float32)
B = np.random.randn(256, 256).astype(np.float32)

# GEMM
C = sk.sgemm(A, B)                     # C = A @ B
sk.sgemm(A, B, C, alpha=2.0, beta=0.5) # C = 2*A@B + 0.5*C  (in-place)

# Activations
y = sk.gelu(x)                          # GeLU (out-of-place)
sk.gelu_inplace(x)                      # GeLU (in-place, no allocation)
y = sk.silu(x)                          # SiLU / Swish
y = sk.relu(x)                          # ReLU
y = sk.softmax(logits, axis=-1)         # Softmax over last axis
y = sk.layer_norm(x, gamma=g, beta=b)  # LayerNorm with affine parameters

# Diagnostics
print(sk.build_info())    # ISA, FMA support, build timestamp
print(sk.detected_isa())  # 'avx2', 'avx512', 'scalar'
```

### C++ (standalone bench)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/bench             # RDTSC cycle-accurate microbenchmark
./build/bench_stat        # Statistical benchmark with CI (30 reps, JSON output)
ctest --test-dir build --output-on-failure
```

### Docker

```bash
docker build -t simd-ml-kernels .
docker run --rm simd-ml-kernels ./bench
```

---

## Performance

Typical single-core numbers on an x86-64 CPU with AVX2/AVX-512 at ~2.1 GHz
(this environment; run your own benchmarks — see below). "Speedup vs Scalar"
compares `sgemm_packed` (the kernel Python's `sk.sgemm()` actually calls)
against a plain triple-loop reference, both compiled with `-O3 -ffast-math`:

| Kernel | Size | Speedup vs Scalar | vs OpenBLAS (1T) |
|--------|------|-------------------|------------------|
| GEMM   | 64×64 | ~0.7× (see note) | ~4% |
| GEMM   | 128×128 | ~10× | ~29% |
| GEMM   | 256×256 | ~26× | ~55% |
| GEMM   | 512×512 | N/A (scalar too slow to measure) | ~59% |
| GeLU   | 1M elements | ~6–10× | N/A |

**Honest note on N=64:** the packed kernel is measurably *slower* than a
plain scalar triple-loop at this size — GCC's `-O3 -ffast-math` auto-vectorizes
the simple reference loop well enough that panel-packing overhead (which the
packed kernel pays regardless of size) makes it a net loss until N≥128. This
is the same small-matrix packing-overhead finding documented in
[`docs/BENCHMARKS.md § Positioning vs OpenBLAS`](docs/BENCHMARKS.md#positioning-vs-openblas-captured-2026-07-ci-environment) — not a bug, but a
concrete argument for the "skip packing below a size threshold" item tracked
in [`docs/ROADMAP.md §v0.9`](docs/ROADMAP.md).

The GEMM kernel auto-dispatches to an AVX-512 8×16 micro-kernel on AVX-512-capable
hardware (falling back to AVX2 8×8 otherwise) — see [`docs/ROADMAP.md §v0.8`](docs/ROADMAP.md).
The gap to OpenBLAS is well-understood and intentional — see [`docs/DESIGN.md §7`](docs/DESIGN.md#7-explicitly-acknowledged-gaps-vs-production-blas). Run `./bench_stat --sizes 64,128,256,512,1024 --reps 30` with `-DBENCH_OPENBLAS=ON` on your hardware for reproducible numbers.

For rigorous comparison with CPU frequency locked:
```bash
sudo cpupower frequency-set -g performance
taskset -c 0 ./build/bench_stat --reps 50 --output results.json
```

---

## Build Requirements

| Requirement | Version |
|-------------|---------|
| C++ Compiler | GCC ≥ 12 or Clang ≥ 15 (C++17) |
| CMake | ≥ 3.22 |
| Python | ≥ 3.9 (for Python extension) |
| pybind11 | ≥ 2.11 (for Python extension) |
| AVX2 + FMA | Required for the SIMD paths |

Optional: `libopenblas-dev` for `./bench_stat` baseline comparisons (`-DBENCH_OPENBLAS=ON`).

---

## Repository Layout

```
simd-ml-kernels/
├── src/
│   ├── kernels/
│   │   ├── avx_matmul.cpp              # 6×16 AVX2 + AVX-512 GEMM
│   │   ├── intrinsic_gelu.cpp          # GeLU (exact erf and tanh variants)
│   │   ├── cache_alloc.hpp             # 64-byte aligned allocator
│   │   ├── gemm/
│   │   │   ├── avx2_gemm_packed.cpp    # 8×8 packed GEMM (Goto/BLIS style)
│   │   │   ├── avx2_gemm_packed.hpp
│   │   │   ├── gemm_dispatcher.cpp     # Runtime ISA dispatch
│   │   │   └── naive_gemm.hpp          # Scalar reference
│   │   └── activations/
│   │       ├── activations.hpp
│   │       ├── relu_avx2.cpp
│   │       ├── silu_avx2.cpp
│   │       ├── softmax_avx2.cpp
│   │       └── layer_norm_avx2.cpp     # NEW: LayerNorm (3-pass, AVX2)
│   ├── dispatch/
│   │   ├── cpuid.hpp                   # Runtime CPU feature detection
│   │   └── kernel_registry.hpp
│   ├── bindings/
│   │   └── pybind_entry.cpp            # Python/NumPy bindings
│   ├── main_bench.cpp                  # RDTSC cycle-accurate bench
│   └── bench_stat.cpp                  # Statistical bench with CI + JSON
├── tests/
│   ├── test_precision.py               # NumPy/PyTorch cross-validation
│   ├── test_bindings_edge_cases.py     # Input validation + error handling
│   ├── test_gemm.cpp                   # C++ GEMM correctness
│   ├── test_gemm_packed.cpp            # C++ packed GEMM correctness
│   ├── test_gelu.cpp                   # C++ GeLU correctness
│   ├── test_activations.cpp            # C++ ReLU/SiLU/Softmax/LayerNorm
│   ├── test_alignment.cpp              # Allocator alignment verification
│   └── test_bench.py                   # Python benchmark + plots
├── benchmarks/
│   ├── check_regression.py             # CI regression gating
│   ├── results/
│   │   ├── bench_results.json          # Committed baseline
│   │   └── README.md
│   └── run_bench.sh
├── docs/
│   ├── DESIGN.md                       # Architecture + design decisions
│   ├── BENCHMARKS.md                   # Captured results + methodology
│   ├── ROADMAP.md
│   └── architecture.md
├── .github/workflows/
│   ├── ci.yml                          # Build + test (every push)
│   ├── build-and-test.yml
│   ├── bench.yml                       # Weekly statistical benchmark
│   └── guardrail.yml                   # Static analysis + security
├── CMakeLists.txt
├── pyproject.toml
├── python/simd_ml_kernels/             # Python package: __init__.py, _core.pyi stubs
├── python/simd_kernels/                # Deprecated import alias
├── Dockerfile
└── README.md
```

---

## Testing

```bash
# C++ unit tests
ctest --test-dir build --output-on-failure

# Python precision tests (validates against NumPy/PyTorch)
pytest tests/test_precision.py -v

# Edge case / error handling tests
pytest tests/test_bindings_edge_cases.py -v

# Quick smoke test
python -c "
import numpy as np, simd_ml_kernels as sk
A = np.random.randn(32, 32).astype(np.float32)
C = sk.sgemm(A, A.T)
x = np.random.randn(1024).astype(np.float32)
y = sk.gelu(x)
print(sk.build_info())
print('All smoke tests passed.')
"
```

---

## CI Pipeline

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Every push / PR | Build (GCC-12, GCC-13, Clang-15) + C++ tests + Python precision tests |
| `build-and-test.yml` | Push to main | Focused build validation |
| `bench.yml` | Weekly (Sunday 4 AM UTC) | `bench_stat` → JSON → regression check (−20% threshold) |
| `guardrail.yml` | Every push / PR | Static analysis + security scan |

---

## Contributing

Issues, pull requests, and benchmark reports from different hardware are
welcome. If you run `bench_stat` on a machine with AVX-512 (e.g. Sapphire
Rapids, Zen 4), please share the results — reproducible numbers from diverse
hardware are the project's most valuable asset.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architectural context before
proposing changes to the hot-loop kernels.

---

## Citation

```bibtex
@misc{myet2026intrinsicml,
  title  = {{IntrinsicML}: Transparent Hand-Vectorized {SIMD} Microkernels
            for Machine Learning Workloads},
  author = {Myet, Min Htet},
  year   = {2026},
  url    = {https://zenodo.org/records/20616502}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
