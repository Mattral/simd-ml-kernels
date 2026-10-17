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
| **SiLU** | `activations/silu_avx2.cpp` | `x * sigmoid(x)` via the exp-based tanh; max abs error ≤ 2e-6 |
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

./build/bench_stat        # C++ wall-clock harness for kernel development
./build/bench             # C++ RDTSC harness
ctest --test-dir build --output-on-failure
```

### Docker

```bash
docker build -t simd-ml-kernels .
docker run --rm simd-ml-kernels ./build/bench_stat
```

---

## Performance

Generated from committed JSON (`benchmarks/results/`) by
`benchmarks/make_tables.py`; protocol, machine details and per-case
interquartile ranges in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Single
thread; the baseline runs in the same process on the same inputs, interleaved
call by call.

<!-- BEGIN GENERATED: results-summary -->
#### lnl-226v-wsl

Source: Intel(R) Core(TM) Ultra 5 226V, WSL2 vCPU 2 (P-core by measured clock, see protocol); Python 3.12.3, GNU 13.3.0, commit `7a698867be89`; baselines: NumPy 2.5.4 with scipy-openblas 0.3.34.106.0, PyTorch 2.14.1+cpu; 30 interleaved reps after 5 warm-up; 2026-10-17T04:53:44+00:00; [`benchmarks/results/lnl-226v-wsl_2026-10-17_7a698867be89.json`](benchmarks/results/lnl-226v-wsl_2026-10-17_7a698867be89.json).

**FP32 GEMM, square, single thread**

| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS |
|---:|---:|---:|---:|
| 128 | 24.4 | 103.5 | 24% |
| 256 | 90.7 | 120.5 | 75% |
| 512 | 100.4 | 114.1 | 88% |
| 1024 | 98.9 | 116.8 | 85% |
| 2048 | 97.7 | 118.3 | 83% |

**Activations and normalization, out-of-place, single thread**

| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch |
|---|---|---:|---:|---:|---:|
| `gelu_exact` | 1048576 | 2.414 | 0.563 | 1.571 | 1.54× |
| `gelu_tanh` | 1048576 | 1.443 | 0.341 | 0.523 | 2.76× |
| `silu` | 1048576 | 1.739 | 0.405 | 1.800 | 0.97× |
| `relu` | 1048576 | 4.485 | 1.063 | 4.194 | 1.07× |
| `softmax` | 1024×1024 | 1.729 | 0.393 | 1.347 | 1.28× |
| `layer_norm` | 1024×1024 | 3.184 | 0.740 | 3.905 | 0.82× |
<!-- END GENERATED: results-summary -->

Run it on your hardware:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 2 python -m simd_ml_kernels bench
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
│   └── test_cli.py                     # selftest / info CLI
├── benchmarks/
│   ├── make_tables.py                  # README + BENCHMARKS tables from results/*.json
│   └── results/                        # committed benchmark JSON (schema in README.md)
├── docs/
│   ├── DESIGN.md                       # Architecture + design decisions
│   ├── BENCHMARKS.md                   # Protocol + generated results
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

# Python tests (float64 references; PyTorch cross-checks run when torch is installed)
pytest tests -v

# Every kernel vs float64 (exit 1 on failure)
python -m simd_ml_kernels selftest```

---

## CI Pipeline

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Every push / PR | Build (GCC-12, GCC-13, Clang-15) + C++ tests + Python precision tests |
| `build-and-test.yml` | Push to main | Focused build validation |
| `bench.yml` | Weekly (Sunday 4 AM UTC) | `python -m simd_ml_kernels bench --json` on a CI runner, uploaded as an artifact (shared runners: not used for claims) |
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
