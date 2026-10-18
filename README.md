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
| **GEMM** (SGEMM) | `gemm/avx2_gemm_packed.cpp`, `gemm/avx512_gemm_packed.cpp` | 6×16 AVX2 (12 accumulators) and 12×32 AVX-512 (24 accumulators) kernels; unpacked for small products, Goto/BLIS 5 loops with block sizes from CPUID leaf 4 otherwise; `trans_a`/`trans_b` handled in packing |
| **FP16 / BF16 GEMM** (`sgemm_f16`, `sgemm_bf16`) | same files; `gemm/avx512bf16_gemm.cpp` | Inputs converted to FP32 once while packing, then the FP32 kernels; FP32 accumulation and output; `vdpbf16ps` on AVX512_BF16 CPUs (speed not measured) |
| **GeLU** | `intrinsic_gelu.cpp` | Exact (erf, rational approximation) by default or tanh formula (`approximate="tanh"`), as in PyTorch; max abs error ≤ 2e-6 |
| **ReLU** | `activations/relu_avx2.cpp` | `_mm256_max_ps`, exact |
| **SiLU** | `activations/silu_avx2.cpp` | `x / (1 + exp(-x))`; error ≤ 1e-35 + 1e-6·\|ref\| for every finite x |
| **Softmax** | `activations/softmax_avx2.cpp` | Numerically stable (max-subtraction) |
| **LayerNorm** | `activations/layer_norm_avx2.cpp` | Two-pass statistics (float lanes, flushed to double), then normalize; optional γ/β; error ≤ 1e-5 + 1e-4·\|ref\| for \|mean\| ≤ 30·std |

Every kernel has a scalar, an AVX2 and (GEMM only) an AVX-512 version, each
compiled separately with its own flags; the one to use is chosen at run time
from CPUID and XGETBV (docs/DESIGN.md §5–6). The extension runs on any x86-64
CPU and, scalar only, on other architectures.

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
y = sk.sgemm(x2d, W, trans_b=True)     # x2d @ W.T, W stored [out, in]; no copy

# Activations
y = sk.gelu(x)                          # GeLU (out-of-place)
sk.gelu_inplace(x)                      # GeLU (in-place, no allocation)
y = sk.silu(x)                          # SiLU / Swish
y = sk.relu(x)                          # ReLU
y = sk.softmax(logits, axis=-1)         # Softmax over last axis
y = sk.layer_norm(x, gamma=g, beta=b)  # LayerNorm with affine parameters

# Diagnostics
print(sk.build_info())    # ISA, FMA support, build timestamp
print(sk.detected_isa())  # kernel table in use: 'avx512', 'avx2' or 'scalar'
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
call by call. The GHz column is the core clock measured during the case;
cases where it fell below the P-core threshold are not shown.

<!-- BEGIN GENERATED: results-summary -->
#### lnl-226v-wsl, 2026-10-18, commit `045918e51839`

Source: Intel(R) Core(TM) Ultra 5 226V, WSL2 vCPU 2 (P-core by measured clock, see protocol); Python 3.12.3, GNU 13.3.0, commit `045918e51839`; baselines: NumPy 2.5.4 with scipy-openblas 0.3.34.106.0, PyTorch 2.14.1+cpu; 30 interleaved reps after 5 warm-up; 2026-10-18T07:17:13+00:00; [`benchmarks/results/lnl-226v-wsl_2026-10-18_045918e51839.json`](benchmarks/results/lnl-226v-wsl_2026-10-18_045918e51839.json).

**FP32 GEMM, square, single thread**

| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS | core GHz median (min–max) |
|---:|---:|---:|---:|---:|
| 128 | 98.7 | 107.7 | 92% | 4.34 (3.80–4.51) |
| 256 | 106.2 | 114.4 | 93% | 4.26 (3.97–4.65) |
| 512 | 118.7 | 117.7 | 101% | 4.46 (3.79–4.59) |
| 1024 | 119.4 | 122.1 | 98% | 4.35 (4.18–4.71) |
| 2048 | 117.4 | 120.9 | 97% | 4.39 (4.19–4.58) |

**FP16 / BF16-input GEMM, FP32 accumulation and output, single thread** (PyTorch: `torch.matmul` on float16 / bfloat16 tensors, which returns float16 / bfloat16)

| input | M×N×K | simd-ml-kernels GFLOP/s | our FP32 GFLOP/s | % of FP32 | PyTorch GFLOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | 128×128×128 | 104.6 | 109.1 | 96% | 71.5 | 1.46× | 4.29 (3.65–4.65) |
| FP16 | 256×256×256 | 100.3 | 105.7 | 95% | 71.1 | 1.41× | 4.13 (3.75–4.50) |
| FP16 | 512×512×512 | 116.1 | 120.7 | 96% | 82.2 | 1.41× | 4.28 (3.95–4.42) |
| FP16 | 1024×1024×1024 | 116.9 | 119.2 | 98% | 82.8 | 1.41× | 4.36 (3.78–4.59) |
| FP16 | 2048×2048×2048 | 116.3 | 116.3 | 100% | 81.8 | 1.42× | 4.43 (3.95–4.57) |
| FP16 | 1×4096×4096 | 5.3 | 2.9 | 180% | 8.4 | 0.63× | 4.49 (4.13–4.63) |
| BF16 | 128×128×128 | 103.6 | 103.8 | 100% | 89.2 | 1.16× | 4.43 (4.18–4.64) |
| BF16 | 256×256×256 | 114.4 | 115.5 | 99% | 113.6 | 1.01× | 4.41 (4.16–4.72) |
| BF16 | 512×512×512 | 120.3 | 120.3 | 100% | 118.5 | 1.02× | 4.17 (4.05–4.54) |
| BF16 | 1024×1024×1024 | 117.2 | 117.9 | 99% | 120.3 | 0.97× | 4.30 (3.93–4.70) |
| BF16 | 2048×2048×2048 | 118.2 | 117.4 | 101% | 121.7 | 0.97× | 4.48 (3.63–4.64) |
| BF16 | 1×4096×4096 | 5.0 | 3.1 | 162% | 8.5 | 0.58× | 4.36 (3.85–4.63) |

**Activations and normalization, out-of-place, single thread**

| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|
| `gelu_exact` | 1048576 | 2.428 | 0.545 | 0.368 | 6.60× | 4.45 (3.70–4.54) |
| `gelu_tanh` | 1048576 | 1.521 | 0.342 | 0.517 | 2.95× | 4.45 (4.18–4.52) |
| `silu` | 1048576 | 2.904 | 0.677 | 2.178 | 1.33× | 4.29 (3.62–4.66) |
| `relu` | 1048576 | 4.994 | 1.127 | 4.439 | 1.13× | 4.43 (3.93–4.54) |
| `softmax` | 1024×1024 | 1.764 | 0.395 | 1.364 | 1.29× | 4.47 (4.24–4.70) |
| `layer_norm` | 1024×1024 | 3.398 | 0.772 | 3.629 | 0.94× | 4.40 (3.70–4.56) |
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
| AVX2 + FMA + F16C | At run time, for the AVX2 kernels (scalar kernels otherwise) |

Optional: `libopenblas-dev` for `./bench_stat` baseline comparisons (`-DBENCH_OPENBLAS=ON`).

---

## Repository Layout

The map of the source tree, the build/test/benchmark commands and the checklist
for adding a kernel are in [AGENTS.md](AGENTS.md); design notes are in
[docs/DESIGN.md](docs/DESIGN.md).

---

## Testing

```bash
# C++ unit tests
ctest --test-dir build --output-on-failure

# Python tests (float64 references; PyTorch cross-checks run when torch is installed)
pytest tests -v

# Every kernel vs float64 (exit 1 on failure)
python -m simd_ml_kernels selftest
```

---

## CI Pipeline

Every pull request builds with GCC 12/13 and Clang 15, runs the C++ tests, the
Python tests with PyTorch cross-checks, `selftest`, stubtest, the examples and
an ASan + UBSan build; the workflows are listed in
[docs/DESIGN.md](docs/DESIGN.md) §8.

---

## Contributing

Issues, pull requests and benchmark results from other hardware are welcome:
run `python -m simd_ml_kernels bench --json` with the protocol in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) and open a PR with the JSON. Start
with [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

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
