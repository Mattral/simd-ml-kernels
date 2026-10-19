# Setup Guide

---

## Install a release

0.3.0rc1, a release candidate, is on TestPyPI; nothing is on PyPI yet
(`pip install simd-ml-kernels` will work after the 0.3.0 release):

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ simd-ml-kernels==0.3.0rc1
python -m simd_ml_kernels selftest
```

Wheels exist for CPython 3.9–3.13 on Linux x86-64 (manylinux_2_28; scalar,
AVX2 and AVX-512 kernels, chosen at run time) and macOS arm64 (scalar kernels
only). Anywhere else (Windows, macOS x86-64, other Python versions) pip builds
the sdist, which needs a C++17 compiler; pip fetches CMake, Ninja and pybind11
itself. A build from the sdist reports `git_commit` as `unknown`.

---

## Prerequisites (building from source)

| Tool | Minimum version | Notes |
|---|---|---|
| CMake | 3.22 | |
| GCC | 12+ | or Clang 15+; C++17. AVX-512 BF16 / VNNI units are built only if the compiler accepts their flags |
| Python | 3.9+ | for the Python extension and tests |
| pybind11 | 2.11+ | via `pip install pybind11[global]` |
| Ninja | any | recommended for faster builds |

On Debian/Ubuntu 22.04+:

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++-12 python3-dev python3-pip
pip install "pybind11[global]>=2.11" "scikit-build-core>=0.6" numpy scipy pytest
```

---

## C++ build (standalone bench and tests)

```bash
git clone https://github.com/Mattral/simd-ml-kernels.git
cd simd-ml-kernels

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# RDTSC harness (reference cycles, development only)
./build/bench

# Statistical harness (warm-up + timed reps, 95% CI, JSON output)
./build/bench_stat --reps 30 --output /tmp/bench_stat.json   # not the published format

# C++ test suite
ctest --test-dir build --output-on-failure
```

### Optional CMake flags

| Flag | Default | Effect |
|---|---|---|
| `-DSIMD_ML_OPENMP=ON` | OFF | Enable OpenMP multi-threaded GEMM |
| `-DBENCH_OPENBLAS=ON` | OFF | Link OpenBLAS for baseline comparison in `bench_stat` |
| `-DSIMD_ML_SANITIZE=ON` | OFF | Instrument every target (kernels included) with ASan + UBSan |
| `-DSIMD_ML_NATIVE=ON` | OFF | `-march=native` on every unit; the build runs only on CPUs like the build machine (local benchmarking only) |
| `-DSIMD_ML_ENABLE_X86=OFF` | ON on x86-64 | Scalar-only build |
| `-DSIMD_ML_BUILD_TESTS=OFF`, `-DSIMD_ML_BUILD_BENCHMARKS=OFF` | ON | Skip the C++ tests / the `bench` and `bench_stat` harnesses |

### Environment variables (run time)

| Variable | Effect |
|---|---|
| `SIMD_ML_ISA=scalar\|avx2\|avx512` | Caps the kernel table chosen at startup, e.g. to test or benchmark AVX2 on an AVX-512 CPU |
| `SIMD_ML_BF16_DOT=1` | Lets `sgemm_bf16` use `vdpbf16ps` on AVX512_BF16 CPUs (off by default; slower where measured) |
| `OMP_NUM_THREADS` | Threads per GEMM call in an OpenMP build, until `set_num_threads()` is called |

---

## Python extension

```bash
# Build and install in editable mode (needs the pip packages above)
pip install -e . --no-build-isolation

# Verify the install: every kernel against a float64 reference, exit code 1 on failure
python -m simd_ml_kernels selftest

# Or, by hand:
python -c "
import simd_ml_kernels, numpy as np
print(simd_ml_kernels.build_info())
A = np.random.randn(64, 64).astype(np.float32)
C = simd_ml_kernels.sgemm(A, A.T.copy())
print('GEMM smoke test: PASS, shape =', C.shape)
x = np.random.randn(1024).astype(np.float32)
y = simd_ml_kernels.gelu(x)
y = simd_ml_kernels.layer_norm(x)
print('Activation smoke tests: PASS')
"
```

`pip install .` (no `-e`) builds in an isolated environment and needs only a
compiler. The tests import the installed package, so reinstall after C++
changes.

### Run Python tests

```bash
python -m simd_ml_kernels selftest   # every kernel vs float64
pytest tests -v
```

---

## AVX-512 without AVX-512 hardware: Intel SDE

The AVX-512 kernels can be checked on any x86-64 CPU under Intel's Software
Development Emulator (download it from Intel; its license does not allow
redistributing it, and Intel's server refuses scripted CI downloads):

```bash
sde64 -skx -- python -m simd_ml_kernels selftest   # scalar, AVX2 and AVX-512 tables
sde64 -spr -- ./build/simd_tests                   # also AVX512_BF16 and AVX512_VNNI
sde64 -nhm -- python -m simd_ml_kernels selftest   # no AVX: the scalar table only
```

Under `-skx`, NumPy's float64 OpenBLAS matmul gave wrong results, so the INT8
tests use exact int64 references.

---

## Docker

A development environment (Ubuntu 22.04, GCC 12, OpenBLAS) is provided:

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
