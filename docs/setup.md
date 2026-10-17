# Setup Guide

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| CMake | 3.22 | Required for `FetchContent` + `CheckCXXCompilerFlag` |
| GCC | 12+ | or Clang 15+; must support C++17 and AVX2 |
| Python | 3.9+ | for the Python extension and tests |
| pybind11 | 2.11+ | via `pip install pybind11[global]` |
| Ninja | any | recommended for faster builds |

On Debian/Ubuntu 22.04+:

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++-12 python3-dev python3-pip
pip install "pybind11[global]>=2.11" numpy scipy pytest
```

---

## C++ build (standalone bench and tests)

```bash
git clone https://github.com/Mattral/simd-ml-kernels.git
cd simd-ml-kernels

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Cycle-accurate RDTSC benchmark
./build/bench

# Statistical benchmark (30-rep wall-clock, 95% CI, JSON output)
./build/bench_stat --reps 30 --output benchmarks/results/ci_results.json

# C++ test suite
ctest --test-dir build --output-on-failure
```

### Optional CMake flags

| Flag | Default | Effect |
|---|---|---|
| `-DSIMD_ML_OPENMP=ON` | OFF | Enable OpenMP multi-threaded GEMM |
| `-DBENCH_OPENBLAS=ON` | OFF | Link OpenBLAS for baseline comparison in `bench_stat` |
| `-DSIMD_ML_SANITIZE=ON` | OFF | Enable AddressSanitizer + UBSan in test binaries |

---

## Python extension

```bash
# Build and install in editable mode (recommended for development)
pip install -e . --no-build-isolation

# Verify the install: every kernel against a float64 reference, exit code 1 on failure
python -m simd_ml_kernels selftest

# Or, by hand:
python -c "
import simd_ml_kernels, numpy as np
print(simd_ml_kernels.build_info())
A = np.random.randn(64, 64).astype(np.float32)
C = simd_ml_kernels.sgemm(A, A.T)
print('GEMM smoke test: PASS, shape =', C.shape)
x = np.random.randn(1024).astype(np.float32)
y = simd_ml_kernels.gelu(x)
y = simd_ml_kernels.layer_norm(x)
print('Activation smoke tests: PASS')
"
```

### Run Python tests

```bash
python -m simd_ml_kernels selftest   # every kernel vs float64
pytest tests -v
```

---

## Docker

A fully pinned development environment is provided:

```bash
docker build -t simd-ml-kernels .
docker run --rm simd-ml-kernels ./build/bench
docker run --rm simd-ml-kernels ctest --test-dir build --output-on-failure
```

---

## Reproducible performance benchmarking

Follow the protocol in [docs/BENCHMARKS.md](BENCHMARKS.md):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 2 python -m simd_ml_kernels bench --json benchmarks/results/<label>_<date>_<commit>.json \
    --label <label> --core "<which core CPU 2 is>" --note "<power settings>"
python benchmarks/make_tables.py
```
