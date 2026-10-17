# ─── simd-ml-kernels — Reproducible Development & Benchmark Environment ──────────
#
# Provides a pinned, reproducible environment for building, testing, and
# benchmarking simd-ml-kernels with GCC-12, CMake ≥ 3.22, pybind11, and OpenBLAS.
#
# Build:
#   docker build -t simd-ml-kernels .
#
# Run tests:
#   docker run --rm simd-ml-kernels
#
# Run benchmarks (statistical harness):
#   docker run --rm simd-ml-kernels ./build/bench_stat --reps 30 --sizes 64,128,256,512
#
# Run Python benchmark:
#   docker run --rm simd-ml-kernels python3 tests/test_bench.py --no-plot
#
# Interactive shell:
#   docker run --rm -it simd-ml-kernels bash

FROM ubuntu:22.04

LABEL maintainer="Min Htet Myet"
LABEL description="simd-ml-kernels SIMD microkernel — build, test, and benchmark environment"
LABEL org.opencontainers.image.source="https://github.com/Mattral/simd-ml-kernels"

# Prevent interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# ─── System packages ──────────────────────────────────────────────────────────
# gcc-12 / g++-12: C++17 with AVX2/FMA support on Ubuntu 22.04
# cmake: ≥ 3.22 required by CMakeLists.txt (FetchContent + CheckCXXCompilerFlag)
# libopenblas-dev: OpenBLAS headers and library for bench_stat baseline comparison
# numactl: provides `numactl --physcpubind` for core pinning in benchmarks
# python3-dev / python3-pip: for pybind11 extension build
# ninja-build: faster parallel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip \
    libopenblas-dev \
    numactl \
    && rm -rf /var/lib/apt/lists/*

# Ensure gcc-12 / g++-12 are the default compilers
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/cc  cc  /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-12 100

# ─── Python dependencies ──────────────────────────────────────────────────────
# pybind11[global]: installs pybind11 CMake support in a location CMake can find
# scikit-build-core: build backend for the Python extension (pyproject.toml)
# numpy: runtime dependency of simd_ml_kernels Python bindings
# scipy: used in test_precision.py for numerical reference comparisons
# pytest: test runner
RUN pip install --no-cache-dir \
    "pybind11[global]>=2.11" \
    "scikit-build-core>=0.6" \
    "numpy>=1.24" \
    "scipy>=1.11" \
    "pytest>=7.4"

# ─── Build the project ────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . .

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DSIMD_ML_OPENMP=OFF \
    -DBENCH_OPENBLAS=ON \
    && cmake --build build --parallel

# Build and install Python extension
RUN pip install --no-cache-dir -e . --no-build-isolation

# ─── Default command: run C++ test suite ─────────────────────────────────────
CMD ["ctest", "--test-dir", "build", "--output-on-failure", "--parallel", "4"]

# ─── Common usage examples ────────────────────────────────────────────────────
# Run C++ tests:
#   docker run --rm simd-ml-kernels
#
# Run RDTSC cycle-accurate bench:
#   docker run --rm simd-ml-kernels ./build/bench
#
# Run statistical bench (JSON output):
#   docker run --rm simd-ml-kernels ./build/bench_stat \
#       --reps 30 --sizes 64,128,256,512 --output /tmp/results.json
#
# Run Python precision tests:
#   docker run --rm simd-ml-kernels pytest tests/test_precision.py -v
#
# Run Python benchmark:
#   docker run --rm simd-ml-kernels python3 tests/test_bench.py --no-plot --quick
