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
| **INT8 GEMM** (`sgemm_u8s8`) | `gemm/avxvnni_gemm.cpp`, `gemm/avx512vnni_gemm.cpp` | uint8 × int8 with per-column scales, FP32 output; `vpdpbusd` (AVX-VNNI, AVX512_VNNI), exact int32 sums per K block; FP32 conversion path without VNNI |
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
#### lnl-226v-wsl, 2026-10-18, commit `19aeaf83cd36`

Source: Intel(R) Core(TM) Ultra 5 226V, WSL2 vCPU 2 (P-core by measured clock, see protocol); Python 3.12.3, GNU 13.3.0, commit `19aeaf83cd36`; baselines: NumPy 2.5.4 with scipy-openblas 0.3.34.106.0, PyTorch 2.14.1+cpu; 30 interleaved reps after 5 warm-up; 2026-10-18T10:43:27+00:00; [`benchmarks/results/lnl-226v-wsl_2026-10-18_19aeaf83cd36.json`](benchmarks/results/lnl-226v-wsl_2026-10-18_19aeaf83cd36.json).

**FP32 GEMM, square, single thread**

| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS | core GHz median (min–max) |
|---:|---:|---:|---:|---:|
| 128 | 87.3 | 93.3 | 94% | 3.70 (3.64–3.94) |
| 256 | 112.9 | 119.1 | 95% | 4.11 (3.78–4.31) |
| 512 | 113.0 | 111.0 | 102% | 4.18 (3.71–4.36) |
| 2048 | 111.8 | 116.5 | 96% | 4.20 (3.78–4.38) |
Not shown: N = 1024 (median 3.70, min 3.45 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**FP16 / BF16-input GEMM, FP32 accumulation and output, single thread** (PyTorch: `torch.matmul` on float16 / bfloat16 tensors, which returns float16 / bfloat16)

| input | M×N×K | simd-ml-kernels GFLOP/s | our FP32 GFLOP/s | % of FP32 | PyTorch GFLOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | 128×128×128 | 96.7 | 103.9 | 93% | 65.7 | 1.47× | 4.26 (3.63–4.44) |
| FP16 | 256×256×256 | 109.9 | 117.9 | 93% | 82.5 | 1.33× | 4.29 (3.65–4.39) |
| FP16 | 512×512×512 | 111.3 | 117.1 | 95% | 81.2 | 1.37× | 4.35 (4.16–4.45) |
| FP16 | 2048×2048×2048 | 113.3 | 113.2 | 100% | 78.7 | 1.44× | 4.27 (4.07–4.33) |
| FP16 | 1×4096×4096 | 5.1 | 3.8 | 136% | 8.1 | 0.63× | 4.15 (4.06–4.34) |
| BF16 | 128×128×128 | 100.5 | 105.8 | 95% | 90.5 | 1.11× | 4.19 (3.86–4.45) |
| BF16 | 256×256×256 | 107.0 | 115.3 | 93% | 113.7 | 0.94× | 4.26 (3.68–4.42) |
| BF16 | 512×512×512 | 112.2 | 115.9 | 97% | 117.8 | 0.95× | 4.29 (4.00–4.44) |
| BF16 | 1024×1024×1024 | 111.1 | 113.6 | 98% | 115.8 | 0.96× | 4.16 (3.61–4.36) |
| BF16 | 2048×2048×2048 | 114.9 | 114.2 | 101% | 117.0 | 0.98× | 4.33 (3.61–4.47) |
| BF16 | 1×4096×4096 | 5.9 | 4.0 | 145% | 8.6 | 0.68× | 4.17 (3.84–4.35) |
Not shown: FP16 1024×1024×1024 (median 3.91, min 3.08 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**INT8 GEMM (uint8 × per-channel int8, FP32 output), x·Wᵀ, single thread**, path `avx_vnni` (PyTorch: `torch.ops.quantized.linear`, prepacked weights, which returns requantized uint8)

| M×N×K | simd-ml-kernels GOP/s | × our FP32 | FP32-conversion path GOP/s | × our FP32 | PyTorch GOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×128×128 | 235.4 | 2.39× | 93.4 | 0.95× | 115.5 | 2.04× | 4.32 (3.90–4.43) |
| 256×256×256 | 365.2 | 3.22× | 110.9 | 0.98× | 181.6 | 2.01× | 4.37 (3.99–4.44) |
| 512×512×512 | 404.8 | 3.50× | 118.6 | 1.03× | 211.9 | 1.91× | 4.39 (3.95–4.45) |
| 1024×1024×1024 | 396.1 | 3.50× | 113.5 | 1.00× | 239.2 | 1.66× | 4.04 (3.76–4.35) |
| 2048×2048×2048 | 415.9 | 3.68× | 112.4 | 1.00× | 246.8 | 1.69× | 4.27 (3.93–4.40) |
| 1×4096×4096 | 17.7 | 1.15× | 7.2 | 0.47× | 38.6 | 0.46× | 4.22 (3.84–4.44) |

**Activations and normalization, out-of-place, single thread**

| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|
| `gelu_tanh` | 1048576 | 1.459 | 0.353 | 0.519 | 2.81× | 4.14 (3.83–4.43) |
| `silu` | 1048576 | 2.963 | 0.691 | 2.088 | 1.42× | 4.29 (3.85–4.36) |
| `relu` | 1048576 | 4.848 | 1.126 | 4.515 | 1.07× | 4.30 (3.60–4.38) |
| `softmax` | 1024×1024 | 1.873 | 0.448 | 1.350 | 1.39× | 4.18 (3.70–4.35) |
| `layer_norm` | 1024×1024 | 3.601 | 0.859 | 4.129 | 0.87× | 4.19 (3.78–4.37) |
Not shown: `gelu_exact` 1048576 (median 3.91, min 3.38 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.
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
