# simd-ml-kernels

[![CI](https://github.com/Mattral/simd-ml-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/simd-ml-kernels/actions/workflows/ci.yml)
[![Weekly Benchmark](https://github.com/Mattral/simd-ml-kernels/actions/workflows/bench.yml/badge.svg)](https://github.com/Mattral/simd-ml-kernels/actions/workflows/bench.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Mattral/simd-ml-kernels/blob/main/LICENSE)
[![Python 3.9–3.13](https://img.shields.io/badge/Python-3.9–3.13-3776ab)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/Preprint-Zenodo-orange)](https://zenodo.org/records/20616502)

simd-ml-kernels is a C++17 library of hand-written x86 SIMD kernels for the
arithmetic at the core of neural-network inference: matrix multiplication,
the matrix-vector product that dominates generating text with a language
model one token at a time, and activation and normalization functions. Each
kernel has a portable C++ version and AVX2 / AVX-512 / VNNI versions chosen at
run time for the CPU, is tested against a float64 reference with a stated
error bound, and is benchmarked against OpenBLAS or PyTorch on the same core
in the same run. You call it from Python on NumPy arrays; the reasons behind
each kernel's design are written next to the code and in
[docs/DESIGN.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/DESIGN.md).

**End to end.** [`examples/llama/`](https://github.com/Mattral/simd-ml-kernels/tree/main/examples/llama)
generates text with Llama-3.2-1B from these kernels and NumPy (a Python loop,
not a runtime), on the same GGUF weight files as llama.cpp; its FP32 output is
token-identical to Hugging Face transformers. Measured against llama.cpp on
the same cores:

<!-- BEGIN GENERATED: llama-summary -->
#### lnl-226v-wsl, 2026-10-21, commit `f4a501dd2db4`

Source: Intel(R) Core(TM) Ultra 5 226V, P-cores 0-3 (taskset -c 0-3), CPUs [0, 1, 2, 3]; meta-llama/Llama-3.2-1B from llama.cpp's own GGUF files (`--pure`), the same bytes on both sides; llama.cpp v0.7.0 (`620d2225cc43`), CPU-only Release, GGML_NATIVE=ON, GGML_OPENMP=ON, GGML_CPU_REPACK=ON; Python 3.12.3, GNU 13.3.0; 2026-10-21T02:31:26+00:00; [`benchmarks/results/lnl-226v-wsl_2026-10-21_f4a501dd2db4_llama.json`](https://github.com/Mattral/simd-ml-kernels/blob/main/benchmarks/results/lnl-226v-wsl_2026-10-21_f4a501dd2db4_llama.json).

**Tokens/s, greedy decode of 128 tokens after a 32-token prompt.** Ours: the examples/llama/ loop, Python over our GEMVs, median of 3 runs. llama.cpp: `llama-bench -p 0 -n 128`, average, at a depth of 32 KV entries like ours. × = ours / llama.cpp: above 1, ours is faster. † = clock below 3.0 GHz on some thread.

| Weights | Threads | Ours tok/s | llama.cpp tok/s (depth 32) | Ours / llama.cpp |
|---|---|---|---|---|
| F32 | 1 | 6.1 | 5.1 | 1.20× |
| F32 | 2 | 10.8 | 8.8 | 1.23× |
| F32 | 4 | 15.2 | 14.3 | 1.06× |
| Q8_0 | 1 | 21.5 | 13.7 | 1.58× |
| Q8_0 | 2 | 35.6 | 21.1 | 1.69× |
| Q8_0 | 4 | 45.9 | 35.5 | 1.29× |
| Q4_0 | 1 | 34.1 | 26.7 | 1.28× |
| Q4_0 | 2 | 54.1 | 47.9 | 1.13× |
| Q4_0 | 4 | 73.8 | 72.1 | 1.02× |


**Where our time goes** (one extra profiled run per row; ms per decode token)

| Weights | Threads | GEMVs | Attention | Norms, RoPE, SiLU | Other (Python/NumPy glue) | Total (profiled) |
|---|---|---|---|---|---|---|
| F32 | 1 | 157.89 ms (97%) | 1.35 ms (1%) | 2.83 ms (2%) | 0.33 ms (0%) | 162.41 ms |
| F32 | 2 | 92.64 ms (96%) | 1.32 ms (1%) | 2.65 ms (3%) | 0.30 ms (0%) | 96.91 ms |
| F32 | 4 | 62.10 ms (94%) | 1.37 ms (2%) | 2.54 ms (4%) | 0.30 ms (0%) | 66.31 ms |
| Q8_0 | 1 | 43.44 ms (92%) | 1.25 ms (3%) | 2.22 ms (5%) | 0.27 ms (1%) | 47.18 ms |
| Q8_0 | 2 | 25.47 ms (89%) | 1.22 ms (4%) | 1.85 ms (6%) | 0.22 ms (1%) | 28.76 ms |
| Q8_0 | 4 | 18.51 ms (85%) | 1.31 ms (6%) | 1.75 ms (8%) | 0.21 ms (1%) | 21.78 ms |
| Q4_0 | 1 | 26.94 ms (90%) | 1.12 ms (4%) | 1.55 ms (5%) | 0.20 ms (1%) | 29.81 ms |
| Q4_0 | 2 | 17.22 ms (85%) | 1.18 ms (6%) | 1.59 ms (8%) | 0.21 ms (1%) | 20.21 ms |
| Q4_0 | 4 | 10.91 ms (80%) | 1.17 ms (9%) | 1.37 ms (10%) | 0.19 ms (1%) | 13.64 ms |


**Perplexity** on `examples/llama/data/alice_ch1-2.txt` (5742 tokens; llama-perplexity's method: chunks of 512, second half scored), ours and `llama-perplexity -c 512` on the same GGUFs

| Weights | Ours PPL | Δ vs our F32 | llama.cpp PPL | Δ vs llama.cpp F32 | Ours − llama.cpp |
|---|---|---|---|---|---|
| F32 | 5.9625 | — | 5.9623 ± 0.2275 | — | +0.0002 |
| Q8_0 | 5.9830 | +0.0205 | 5.9782 ± 0.2281 | +0.0159 | +0.0048 |
| Q4_0 | 8.8892 | +2.9267 | 8.8716 ± 0.3637 | +2.9093 | +0.0176 |
<!-- END GENERATED: llama-summary -->

**Preprint**: [IntrinsicML: Transparent Hand-Vectorized SIMD Microkernels for ML Workloads](https://zenodo.org/records/20616502)

## Who it is for

Use it if you want to:

- read how a matrix multiply or a memory-bound matrix-vector product gets
  close to the hardware's limit, with the measurement behind each choice;
- compute decode-shaped `x @ W.T` (one row of activations times a weight
  matrix) from NumPy, on FP32, FP16 or BF16 weights;
- test or compare SIMD code paths (scalar, AVX2, AVX-512) on one machine:
  `sgemm(..., isa=...)`, `SIMD_ML_ISA`, Intel SDE.

Do not use it if you just want a model to run fast on a CPU: use PyTorch
(oneDNN), a BLAS such as OpenBLAS or MKL, or llama.cpp. This library is
single-threaded by default, has no autograd, no model runtime and no batching
logic, vectorizes only on x86-64, and loses to those libraries in several
measured cases (Known Honest Limitations below). It is pre-1.0 (0.3.0).

## Install

```bash
pip install simd-ml-kernels
python -m simd_ml_kernels selftest   # every kernel vs a float64 reference; exits 1 on failure
```

Wheels exist for CPython 3.9–3.13 on Linux x86-64 (manylinux_2_28) and macOS
arm64 (scalar kernels only); anywhere else pip builds from source, which needs
a C++17 compiler. From a clone: `pip install .`. Details:
[docs/setup.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/setup.md).

```python
import numpy as np
import simd_ml_kernels as sk

x = np.random.randn(1, 4096).astype(np.float16)      # one token's activations
W = np.random.randn(11008, 4096).astype(np.float16)  # weights [out, in], as torch.nn.Linear stores them
y = sk.sgemm_f16(x, W, trans_b=True)                 # y = x @ W.T in float32, W read in place: the decode GEMV
h = sk.silu(y)                                       # activation
A = np.random.randn(512, 512).astype(np.float32)
C = sk.sgemm(A, A)                                   # FP32 matrix multiply, C = A @ A
print(sk.detected_isa(), sk.build_info()["sgemm_blocking"])
```

## What's measured

Every number below and in the
[Performance](https://github.com/Mattral/simd-ml-kernels#performance) tables
comes from a
JSON file in
[benchmarks/results/](https://github.com/Mattral/simd-ml-kernels/tree/main/benchmarks/results),
written by `python -m simd_ml_kernels bench`: one pinned core, one thread, the
baseline in the same process on the same inputs, alternating call by call.
All runs, with interquartile ranges, are in
[docs/BENCHMARKS.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/BENCHMARKS.md).
The terms used there:

- **GEMM** (general matrix multiply, C = A·B): the operation behind most of a
  neural network's arithmetic when many inputs are processed together. Each
  number loaded is used many times, so the limit is arithmetic, measured in
  GFLOP/s (billions of floating-point operations per second).
- **% of OpenBLAS**: our GEMM's GFLOP/s divided by OpenBLAS's (through
  NumPy's `matmul`) on the same core in the same run; 100% is level.
- **GEMV** (matrix times one vector): generating one token multiplies a single
  row of activations by every weight matrix of the model, so each weight is
  read once and used once. The core spends its time waiting for memory, and
  speed is measured in GB/s of data read.
- **% of read roof**: the GEMV's GB/s divided by the fastest read-only loop
  the same core ran in the same case (timed alongside it): how close it gets
  to what one core can read from memory (a lower bound on that limit). The STREAM triad (two arrays read, one written) is
  shown as a second reference.
- **Q8_0**: llama.cpp/ggml's 8-bit weight format: blocks of 32 int8 values
  with one FP16 scale, 34 bytes per 32 weights (1.0625 bytes each).
- **Q4_0**: ggml's 4-bit weight format: blocks of 32 4-bit values (two per
  byte) with one FP16 scale, 18 bytes per 32 weights (0.5625 bytes each).
- **FP16 / BF16 / INT8**: weight formats of 2, 2 and 1 bytes instead of
  FP32's 4. FP16 is IEEE half precision; BF16 keeps FP32's exponent range with
  8 bits of precision; INT8 stores integers with one scale per output column.
  Inputs are converted exactly and summed in FP32 (exact int32 sums for INT8),
  so the gain is fewer bytes to read, not cheaper arithmetic, except INT8 with
  VNNI instructions.

Summary:

- FP32 GEMM: 93–99% of OpenBLAS at N = 128–2048 on the reference laptop's
  P-core (`lnl-226v-wsl_2026-10-18_ecc219baa1f2`; N = 1024 not shown, clock
  below the threshold); 87–107% on AVX-512, a 2-vCPU Emerald Rapids cloud VM
  (`cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31`).
- Decode GEMV (laptop, `ecc219baa1f2`): FP32 at 97–99% of the read roof,
  1.03–1.05× PyTorch's `F.linear` and 1.00–1.20× OpenBLAS; FP16/BF16 at
  90–99%, 1.78–1.92× PyTorch's `F.linear` on the same 16-bit weights. Not
  measured on AVX-512.
- Q8_0 decode GEMV (laptop, `290b91baad26`): 92–95% of the read roof,
  1.46–1.82× ggml's Q8_0 MUL_MAT (llama.cpp v0.7.0, one thread), 3.49–3.61×
  our FP32 GEMV.
- Q4_0 decode GEMV (laptop, `42f64c7d76b3`): 68–83% of the read roof,
  1.58–2.19× ggml's Q4_0 MUL_MAT (llama.cpp v0.7.0, one thread), 1.38–1.67×
  our Q8_0 GEMV.
- Threaded decode GEMV (OpenMP build, laptop's 4 P-cores, `38cd710929a3`):
  at 4 threads 1.22–2.22× (Q4_0, every shape) and 1.23–1.82× (Q8_0, level
  at 512×2048) ggml's MUL_MAT at the same thread count, on the same
  weights; Llama-3.2-1B GEMV time per token at 4 threads 10.4 ms (Q4_0) and
  17.0 ms (Q8_0) against ggml's 14.5 and 21.6. Four P-cores read only
  2.4–3.0× one core's bandwidth, so 4 threads are not 4× faster.
- INT8 GEMM with AVX-VNNI (laptop, `ecc219baa1f2`): 3.36–3.60× our FP32 GEMM
  and 1.72–2.19× PyTorch's quantized linear at 256³ and 1024³.
- Activations (laptop, `ecc219baa1f2`): 1.05–2.97× PyTorch, except LayerNorm
  at 0.92×.

## Known Honest Limitations

Where it loses or has not been measured (the full table, with planned fixes,
is in
[docs/ROADMAP.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/ROADMAP.md#known-honest-limitations-never-remove-this-section)):

- One row times a 4096×4096 INT8 matrix on the cloud VM: 0.13× PyTorch
  (`255f27755c31`, before the row path; on the laptop it is now 1.36×,
  `290b91baad26`).
- On the Emerald Rapids VM (`255f27755c31`): BF16 GEMM at 0.14–0.72× PyTorch,
  whose rate implies AMX tiles, which this library does not use; INT8 GEMM at
  0.67–0.75× PyTorch from N = 512 up; FP32 GEMM at 87% of OpenBLAS at
  N = 2048.
- LayerNorm at 0.92× PyTorch on the laptop (`ecc219baa1f2`).
- FP16/BF16 GEMM on AVX2 saves memory but is no faster than FP32 on
  compute-bound shapes (AVX2 has no 16-bit FMA).
- Single-threaded by default: OpenMP (GEMM and decode GEMV) is a build
  option. GEMV scaling is measured only on the laptop's 4 P-cores under
  WSL2, where 4 cores read 2.4–3.0× one core's bandwidth; at 4 threads the
  smallest Llama-3.2-1B matrix in Q8_0 (512×2048) is level with ggml. The
  INT8 GEMM has no threaded path.
- AVX-512 is measured only on a cloud VM whose clock and neighbours are not
  controlled; the AVX-512 GEMV is not measured.

---

## Kernels

| Kernel | Implementation | Notes |
|--------|---------------|-------|
| **GEMM** (SGEMM) | `gemm/avx2_gemm_packed.cpp`, `gemm/avx512_gemm_packed.cpp` | 6×16 AVX2 (12 accumulators) and 12×32 AVX-512 (24 accumulators) kernels; unpacked for small products, Goto/BLIS 5 loops with block sizes from CPUID leaf 4 otherwise; `trans_a`/`trans_b` handled in packing |
| **FP16 / BF16 GEMM** (`sgemm_f16`, `sgemm_bf16`) | same files; `gemm/avx512bf16_gemm.cpp` | Inputs converted to FP32 once while packing, then the FP32 kernels; FP32 accumulation and output; `vdpbf16ps` path opt-in (`SIMD_ML_BF16_DOT=1`; slower than conversion where measured) |
| **Decode GEMV** (`x @ W.T`: `sgemm` / `sgemm_f16` / `sgemm_bf16` with `trans_b=True`, ≤ 8 rows) | same files (`dot_rows`) | W [N, K] read in place, 4 (FP32) or 8 (FP16/BF16) rows at a time, 16-bit weights converted in registers; bound by memory bandwidth, reported as GB/s and % of the measured read stream and STREAM triad |
| **INT8 GEMM** (`sgemm_u8s8`) | `gemm/avxvnni_gemm.cpp`, `gemm/avx512vnni_gemm.cpp`, `gemm/int8_rows.hpp` | uint8 × int8 with per-column scales, FP32 output; `vpdpbusd` (AVX-VNNI, AVX512_VNNI), exact int32 sums per K block; FP32 conversion path without VNNI; one row with `trans_b` reads B in place |
| **Q8_0 decode GEMV** (`quantize_q8_0`, `gemv_q8_0`) | `quant/q8_0_scalar.cpp`, `quant/q8_0_rows.hpp` | Weights in ggml's `block_q8_0` (byte-identical; 34 bytes per 32), x quantized to Q8 once per call; exact integer block dots (`vpdpbusd` with ggml's sign trick, `vpmaddubsw` without VNNI), FP32 sums; 8 rows read in place, next rows prefetched |
| **Q4_0 decode GEMV** (`quantize_q4_0`, `gemv_q4_0`) | `quant/q4_0_scalar.cpp`, `quant/q4_0_rows.hpp` | Weights in ggml's `block_q4_0` (byte-identical; 18 bytes per 32), x quantized to Q8 once per call; two rows per register, nibbles as `vpdpbusd`'s unsigned operand with −8·Σx as its start (no sign trick), FP32 sums; 8 rows read in place, next rows prefetched |
| **GeLU** | `intrinsic_gelu.cpp` | Exact (erf, rational approximation) by default or tanh formula (`approximate="tanh"`), as in PyTorch; max abs error ≤ 2e-6 |
| **ReLU** | `activations/relu_avx2.cpp` | `_mm256_max_ps`, exact |
| **SiLU** | `activations/silu_avx2.cpp` | `x / (1 + exp(-x))`; error ≤ 1e-35 + 1e-6·\|ref\| for every finite x |
| **Softmax** | `activations/softmax_avx2.cpp` | Numerically stable (max-subtraction) |
| **LayerNorm** | `activations/layer_norm_avx2.cpp` | Two-pass statistics (float lanes, flushed to double), then normalize; optional γ/β; error ≤ 1e-5 + 1e-4·\|ref\| for \|mean\| ≤ 30·std |

Every kernel has a scalar, an AVX2 and (GEMM only) an AVX-512 version, each
compiled separately with its own flags; the one to use is chosen at run time
from CPUID and XGETBV (docs/DESIGN.md §5–6). The extension runs on any x86-64
CPU and, scalar only, on other architectures. The full Python API, with each
function's error bound, is in
[docs/api/python_api.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/api/python_api.md).

---

## Building from source

### Python

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
x = np.random.randn(64, 256).astype(np.float32)

# GEMM
C = sk.sgemm(A, B)                     # C = A @ B
sk.sgemm(A, B, C, alpha=2.0, beta=0.5) # C = 2*A@B + 0.5*C  (in-place)
y = sk.sgemm(x, B, trans_b=True)       # x @ B.T, B stored [out, in]; no copy

# Activations
y = sk.gelu(x)                          # GeLU (out-of-place)
sk.gelu_inplace(x)                      # GeLU (in-place, no allocation)
y = sk.silu(x)                          # SiLU / Swish
y = sk.relu(x)                          # ReLU
y = sk.softmax(x, axis=-1)              # Softmax over last axis
y = sk.layer_norm(x, gamma=np.ones(256, np.float32), beta=np.zeros(256, np.float32))

# Diagnostics
print(sk.build_info())    # version, commit, compiled and usable ISAs, flags, block sizes
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
interquartile ranges in
[docs/BENCHMARKS.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/BENCHMARKS.md).
Single thread; the baseline runs in the same process on the same inputs,
interleaved call by call. The GHz column is the core clock measured during the
case; cases where it fell below the P-core threshold are not shown.

<!-- BEGIN GENERATED: results-summary -->
#### lnl-226v-wsl, 2026-10-20, commit `42f64c7d76b3`

Source: Intel(R) Core(TM) Ultra 5 226V, WSL2 vCPU 2 (P-core by measured clock, see protocol); Python 3.12.3, GNU 13.3.0, commit `42f64c7d76b3`; baselines: NumPy 2.5.4 with scipy-openblas 0.3.34.106.0, PyTorch 2.14.1+cpu; 30 interleaved reps after 5 warm-up; 2026-10-20T02:58:46+00:00; [`benchmarks/results/lnl-226v-wsl_2026-10-20_42f64c7d76b3.json`](https://github.com/Mattral/simd-ml-kernels/blob/main/benchmarks/results/lnl-226v-wsl_2026-10-20_42f64c7d76b3.json).

**FP32 GEMM, square, single thread**

| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS | core GHz median (min–max) |
|---:|---:|---:|---:|---:|
| 128 | 97.1 | 103.8 | 94% | 4.14 (3.74–4.37) |
| 256 | 104.0 | 115.9 | 90% | 4.27 (4.00–4.43) |
| 512 | 116.5 | 121.1 | 96% | 4.41 (3.63–4.43) |
| 1024 | 117.6 | 119.1 | 99% | 4.15 (3.87–4.42) |
| 2048 | 114.0 | 118.7 | 96% | 4.16 (4.02–4.43) |

**FP16 / BF16-input GEMM, FP32 accumulation and output, single thread** (PyTorch: `torch.matmul` on float16 / bfloat16 tensors, which returns float16 / bfloat16)

| input | M×N×K | simd-ml-kernels GFLOP/s | our FP32 GFLOP/s | % of FP32 | PyTorch GFLOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | 128×128×128 | 89.4 | 95.2 | 94% | 59.1 | 1.51× | 4.33 (3.69–4.39) |
| FP16 | 256×256×256 | 109.7 | 116.5 | 94% | 78.6 | 1.40× | 4.22 (3.66–4.44) |
| FP16 | 512×512×512 | 113.5 | 119.7 | 95% | 79.5 | 1.43× | 4.34 (3.69–4.43) |
| FP16 | 1024×1024×1024 | 112.9 | 113.0 | 100% | 79.6 | 1.42× | 4.28 (3.76–4.43) |
| FP16 | 2048×2048×2048 | 113.7 | 112.8 | 101% | 78.1 | 1.46× | 4.19 (3.94–4.42) |
| BF16 | 128×128×128 | 100.7 | 101.3 | 99% | 86.4 | 1.16× | 3.96 (3.75–4.38) |
| BF16 | 256×256×256 | 112.2 | 115.8 | 97% | 113.2 | 0.99× | 4.19 (4.04–4.42) |
| BF16 | 512×512×512 | 117.0 | 118.7 | 99% | 119.2 | 0.98× | 4.14 (3.92–4.43) |
| BF16 | 1024×1024×1024 | 113.2 | 114.7 | 99% | 116.5 | 0.97× | 4.09 (3.79–4.42) |
| BF16 | 2048×2048×2048 | 113.6 | 112.7 | 101% | 115.0 | 0.99× | 4.20 (3.97–4.42) |
Not shown: FP16 1×4096×4096 (median 4.12, min 3.39 GHz, 3 attempts), BF16 1×4096×4096 (median 4.21, min 2.85 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**INT8 GEMM (uint8 × per-channel int8, FP32 output), x·Wᵀ, single thread**, path `avx_vnni` (PyTorch: `torch.ops.quantized.linear`, prepacked weights, which returns requantized uint8)

| M×N×K | simd-ml-kernels GOP/s | × our FP32 | FP32-conversion path GOP/s | × our FP32 | PyTorch GOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256×256×256 | 381.5 | 3.26× | 116.1 | 0.99× | 182.9 | 2.09× | 4.19 (3.77–4.39) |
| 512×512×512 | 398.3 | 3.58× | 109.2 | 0.98× | 204.2 | 1.95× | 4.03 (3.77–4.18) |
| 2048×2048×2048 | 407.7 | 3.64× | 111.8 | 1.00× | 242.2 | 1.68× | 4.33 (3.63–4.44) |
| 1×4096×4096 | 60.0 | 3.61× | 65.8 | 3.96× | 37.3 | 1.61× | 4.30 (3.76–4.39) |
Not shown: 128×128×128 (median 4.22, min 2.49 GHz, 3 attempts), 1024×1024×1024 (median 4.07, min 3.60 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**Decode GEMV y = x·Wᵀ (one row x, W [N, K] as torch.nn.Linear stores it), FP32 output, single thread**; read roof: the read stream timed in each case, interleaved with it; STREAM triad of the run: single-core read stream 32.2 GB/s, STREAM triad 29.0 GB/s (all-core: read not measured, triad not measured). GB/s counts the weights, x and y. (PyTorch: `F.linear` on the same weights in their dtype, which returns that dtype; OpenBLAS: NumPy's matrix-vector product, FP32 only)

| weights | M×N×K | simd-ml-kernels µs | GB/s | % of read roof | % of triad | PyTorch GB/s | ours / PyTorch | OpenBLAS sgemv GB/s | ours / OpenBLAS | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 1×2048×8192 | 1989 | 33.8 | 93% | 117% | 33.2 | 1.02× | 30.7 | 1.10× | 4.40 (3.69–4.44) |
| FP32 | 1×4096×4096 | 1998 | 33.6 | 96% | 116% | 32.3 | 1.04× | 33.3 | 1.01× | 4.20 (3.99–4.39) |
| FP32 | 1×4096×11008 | 5252 | 34.4 | 96% | 119% | 34.5 | 1.00× | 30.0 | 1.14× | 4.09 (3.74–4.27) |
| FP16 | 1×2048×8192 | 1101 | 30.5 | 86% | 105% | 17.6 | 1.74× | — | — | 4.22 (3.96–4.43) |
| FP16 | 1×4096×4096 | 1097 | 30.6 | 85% | 106% | 19.1 | 1.60× | — | — | 4.17 (3.96–4.37) |
| FP16 | 1×4096×11008 | 2872 | 31.4 | 95% | 108% | 17.8 | 1.76× | — | — | 4.15 (3.95–4.40) |
| BF16 | 1×4096×4096 | 1178 | 28.5 | 82% | 98% | 17.7 | 1.61× | — | — | 4.33 (4.12–4.41) |
| BF16 | 1×4096×11008 | 2928 | 30.8 | 87% | 106% | 16.9 | 1.83× | — | — | 4.13 (3.65–4.33) |
Not shown: BF16 1×2048×8192 (median 4.09, min 3.39 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**Q8_0 decode GEMV** (`gemv_q8_0`: ggml block_q8_0 weights, 34 bytes per 32, x quantized to Q8 inside), path `avx_vnni`, single thread; GB/s counts the weights, x and y; × columns: time of that GEMV / ours, in the same interleaved case. PyTorch: FP32 `F.linear` on the dequantized weights, not a quantized kernel. ggml: llama.cpp v0.7.0 (`620d222`), `test-backend-ops perf` MUL_MAT q8_0 × f32, one thread (OMP_THREAD_LIMIT=1, GGML_OPENMP=ON, GGML_NATIVE=ON), median of 3 runs of its own mean over ≥ 1 s on the same weights (not interleaved).

| M×N×K | simd-ml-kernels µs | GB/s | % of read roof | × our FP32 GEMV | × our BF16 GEMV | PyTorch FP32 µs | ours / PyTorch | ggml µs (own timing) | ggml GB/s | ggml % of roof | ours / ggml | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×2048×8192 | 588 | 30.4 | 85% | 3.45× | 1.90× | 2084 | 3.55× | 909 | 19.6 | 55% | 1.55× | 4.25 (3.82–4.41) |
| 1×4096×4096 | 564 | 31.6 | 89% | 3.59× | 2.09× | 2140 | 3.79× | 923 | 19.3 | 55% | 1.64× | 4.22 (4.10–4.44) |
| 1×4096×11008 | 1726 | 27.8 | 81% | 3.24× | 1.75× | 5355 | 3.10× | 2802 | 17.1 | 50% | 1.62× | 4.19 (4.02–4.40) |

**Q4_0 decode GEMV** (`gemv_q4_0`: ggml block_q4_0 weights, 18 bytes per 32, x quantized to Q8 inside), path `avx_vnni`, single thread; GB/s counts the weights, x and y; × columns: time of that GEMV / ours, in the same interleaved case (our Q8_0 GEMV on the Q8_0 blocks of the same W). PyTorch: FP32 `F.linear` on the dequantized weights, not a quantized kernel. ggml: llama.cpp v0.7.0 (`620d222`), `test-backend-ops perf` MUL_MAT q4_0 × f32, one thread (OMP_THREAD_LIMIT=1, GGML_OPENMP=ON, GGML_NATIVE=ON), median of 3 runs of its own mean over ≥ 1 s on the same weights (not interleaved).

| M×N×K | simd-ml-kernels µs | GB/s | % of read roof | × our Q8_0 GEMV | × our FP32 GEMV | PyTorch FP32 µs | ours / PyTorch | ggml µs (own timing) | ggml GB/s | ggml % of roof | ours / ggml | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×2048×8192 | 397 | 23.9 | 68% | 1.38× | 5.12× | 2040 | 5.14× | 628 | 15.1 | 43% | 1.58× | 4.25 (4.08–4.44) |
| 1×4096×4096 | 383 | 24.7 | 70% | 1.45× | 5.40× | 2136 | 5.58× | 610 | 15.5 | 44% | 1.59× | 4.29 (3.94–4.41) |
| 1×4096×11008 | 890 | 28.6 | 83% | 1.67× | 6.05× | 5361 | 6.03× | 1948 | 13.0 | 38% | 2.19× | 4.29 (3.90–4.40) |

**Activations and normalization, out-of-place, single thread**

| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|
| `gelu_exact` | 1048576 | 2.777 | 0.669 | 1.611 | 1.72× | 4.15 (3.74–4.39) |
| `gelu_tanh` | 1048576 | 1.514 | 0.376 | 0.526 | 2.88× | 4.03 (3.73–4.42) |
| `silu` | 1048576 | 2.619 | 0.622 | 1.973 | 1.33× | 4.21 (3.77–4.42) |
| `relu` | 1048576 | 4.304 | 1.010 | 4.430 | 0.97× | 4.26 (3.76–4.40) |
| `softmax` | 1024×1024 | 1.504 | 0.351 | 1.365 | 1.10× | 4.28 (3.83–4.42) |
| `layer_norm` | 1024×1024 | 3.980 | 0.965 | 4.128 | 0.96× | 4.13 (3.72–4.38) |

#### cloud-xeon-emr-2vcpu, 2026-10-18, commit `255f27755c31`

Source: Intel(R) Xeon(R) Processor @ 2.10GHz, vCPU 1 (KVM guest, 2 vCPU; host core not visible); Python 3.13.16, GNU 13.3.0, commit `255f27755c31`; baselines: NumPy 2.5.4 with scipy-openblas 0.3.34.106.0, PyTorch 2.14.1+cu130; 30 interleaved reps after 5 warm-up; 2026-10-18T12:03:45+00:00; [`benchmarks/results/cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31.json`](https://github.com/Mattral/simd-ml-kernels/blob/main/benchmarks/results/cloud-xeon-emr-2vcpu_2026-10-18_255f27755c31.json).

**FP32 GEMM, square, single thread**

| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS | core GHz median (min–max) |
|---:|---:|---:|---:|---:|
| 128 | 95.4 | 89.6 | 107% | 3.08 (2.90–3.22) |
| 256 | 111.0 | 112.6 | 99% | 3.11 (2.93–3.24) |
| 512 | 105.6 | 108.3 | 98% | 3.03 (1.13–3.20) |
| 1024 | 101.7 | 111.8 | 91% | 3.29 (2.82–3.59) |
| 2048 | 108.1 | 124.8 | 87% | 3.18 (3.08–3.68) |

**FP16 / BF16-input GEMM, FP32 accumulation and output, single thread** (PyTorch: `torch.matmul` on float16 / bfloat16 tensors, which returns float16 / bfloat16)

| input | M×N×K | simd-ml-kernels GFLOP/s | our FP32 GFLOP/s | % of FP32 | PyTorch GFLOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | 128×128×128 | 86.3 | 94.9 | 91% | 69.0 | 1.25× | 3.15 (3.00–3.22) |
| FP16 | 256×256×256 | 96.5 | 104.1 | 93% | 95.2 | 1.01× | 3.17 (2.90–3.22) |
| FP16 | 512×512×512 | 110.8 | 109.6 | 101% | 123.3 | 0.90× | 3.18 (2.96–3.20) |
| FP16 | 1024×1024×1024 | 114.9 | 114.9 | 100% | 128.6 | 0.89× | 3.17 (3.02–3.22) |
| FP16 | 2048×2048×2048 | 104.7 | 104.9 | 100% | 117.4 | 0.89× | 3.05 (2.88–3.38) |
| FP16 | 1×4096×4096 | 2.7 | 1.7 | 153% | 4.7 | 0.57× | 3.17 (3.05–3.33) |
| BF16 | 128×128×128 | 100.2 | 99.9 | 100% | 139.1 | 0.72× | 3.06 (2.84–3.20) |
| BF16 | 256×256×256 | 97.8 | 95.2 | 103% | 225.0 | 0.43× | 3.11 (2.93–3.17) |
| BF16 | 512×512×512 | 115.0 | 110.3 | 104% | 640.5 | 0.18× | 3.13 (3.06–3.21) |
| BF16 | 1024×1024×1024 | 112.9 | 113.9 | 99% | 805.2 | 0.14× | 3.18 (3.01–3.44) |
| BF16 | 2048×2048×2048 | 108.7 | 109.0 | 100% | 771.3 | 0.14× | 3.21 (2.87–3.91) |
| BF16 | 1×4096×4096 | 2.3 | 1.7 | 137% | 4.0 | 0.58× | 3.59 (2.97–3.97) |

**INT8 GEMM (uint8 × per-channel int8, FP32 output), x·Wᵀ, single thread**, path `avx512_vnni` (PyTorch: `torch.ops.quantized.linear`, prepacked weights, which returns requantized uint8)

| M×N×K | simd-ml-kernels GOP/s | × our FP32 | FP32-conversion path GOP/s | × our FP32 | PyTorch GOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×128×128 | 114.2 | 1.55× | 73.1 | 0.99× | 46.4 | 2.46× | 3.14 (3.00–3.87) |
| 256×256×256 | 198.6 | 1.99× | 78.5 | 0.79× | 157.8 | 1.26× | 3.22 (3.15–3.31) |
| 512×512×512 | 249.4 | 2.32× | 80.7 | 0.75× | 350.1 | 0.71× | 3.23 (3.17–3.37) |
| 1024×1024×1024 | 229.5 | 2.69× | 66.0 | 0.77× | 306.4 | 0.75× | 3.11 (2.85–3.32) |
| 2048×2048×2048 | 291.9 | 2.83× | 69.9 | 0.68× | 435.2 | 0.67× | 3.17 (2.75–4.01) |
| 1×4096×4096 | 2.7 | 0.38× | 4.1 | 0.57× | 20.1 | 0.13× | 3.32 (3.19–3.38) |

**Activations and normalization, out-of-place, single thread**

| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|
| `gelu_exact` | 1048576 | 1.983 | 0.585 | 1.971 | 1.01× | 3.39 (3.15–3.45) |
| `gelu_tanh` | 1048576 | 1.156 | 0.339 | 0.460 | 2.52× | 3.41 (3.30–3.50) |
| `silu` | 1048576 | 1.712 | 0.542 | 1.704 | 1.00× | 3.16 (2.95–3.20) |
| `relu` | 1048576 | 2.725 | 0.865 | 2.704 | 1.01× | 3.15 (3.00–3.21) |
| `softmax` | 1024×1024 | 1.084 | 0.336 | 1.260 | 0.86× | 3.22 (3.15–3.26) |
| `layer_norm` | 1024×1024 | 2.099 | 0.663 | 2.180 | 0.96× | 3.16 (3.04–3.30) |
<!-- END GENERATED: results-summary -->

### Threaded decode GEMV

OpenMP builds only (`-DSIMD_ML_OPENMP=ON`, off by default). The process runs
on the laptop's 4 P-cores (`taskset -c 0-3`, the 4 low-power E-cores excluded);
ours, PyTorch and ggml use the same thread count, and every percentage is of
the read roof measured at that thread count in the same case.

<!-- BEGIN GENERATED: threads-summary -->
#### lnl-226v-wsl, 2026-10-20, commit `38cd710929a3`, 1/2/4 threads

Source: Intel(R) Core(TM) Ultra 5 226V, WSL2 vCPUs 0-3 (the 4 P-cores; the 4 LP E-cores excluded), 1/2/4 threads; Python 3.12.3, GNU 13.3.0, commit `38cd710929a3`; baselines: NumPy 2.5.4 with scipy-openblas 0.3.34.106.0, PyTorch 2.14.1+cpu; 30 interleaved reps after 5 warm-up; 2026-10-20T08:58:42+00:00; [`benchmarks/results/lnl-226v-wsl_2026-10-20_38cd710929a3_threads.json`](https://github.com/Mattral/simd-ml-kernels/blob/main/benchmarks/results/lnl-226v-wsl_2026-10-20_38cd710929a3_threads.json); CPUs [0, 1, 2, 3].

**Q8_0 / Q4_0 decode GEMV at 1–4 threads**, path `avx_vnni`. GB/s counts the weights, x and y; % of the read roof measured at the same thread count in the same case; scaling = GB/s at N threads / (N × GB/s at 1 thread); ours with weights rotated past the caches, and on the same weights every call (as ggml runs). ggml: llama.cpp v0.7.0 (`620d222`), `test-backend-ops perf` MUL_MAT q8_0 or q4_0 × f32, the same thread count (OMP_THREAD_LIMIT = threads, GGML_OPENMP=ON, GGML_NATIVE=ON), median of 3 runs of its own mean over ≥ 1 s on the same weights (not interleaved).

| format | N×K (Llama-3.2-1B) | threads | simd-ml-kernels µs | GB/s | % of read roof | scaling | same weights µs | ggml µs (own timing) | ggml % of roof | ours / ggml (same weights) | ours / ggml (rotated) | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 512×2048 (k_proj, v_proj) | 1 | 42 | 26.9 | 78% | 100% | 30 | 30 | 110% | 0.98× | 0.71× | 4.48 (4.00–4.58) |
| Q8_0 | 512×2048 (k_proj, v_proj) | 2 | 24 | 47.2 | 71% | 88% | 17 | 18 | 95% | 1.07× | 0.75× | 4.26 (4.04–4.50) |
| Q8_0 | 512×2048 (k_proj, v_proj) | 4 | 18 | 61.4 | 68% | 57% | 12 | 11 | 110% | 0.97× | 0.62× | 3.56 (3.13–3.65) |
| Q8_0 | 2048×2048 (q_proj, o_proj) | 1 | 150 | 29.7 | 81% | 100% | 109 | 128 | 95% | 1.18× | 0.85× | 4.23 (3.95–4.57) |
| Q8_0 | 2048×2048 (q_proj, o_proj) | 2 | 85 | 52.4 | 80% | 88% | 53 | 73 | 93% | 1.38× | 0.86× | 4.11 (3.67–4.52) |
| Q8_0 | 2048×2048 (q_proj, o_proj) | 4 | 60 | 74.8 | 79% | 63% | 36 | 45 | 107% | 1.23× | 0.75× | 3.57 (3.16–3.66) |
| Q8_0 | 2048×8192 (down_proj) | 1 | 504 | 35.4 | 93% | 100% | 474 | 877 | 54% | 1.85× | 1.74× | 4.50 (3.93–4.90) |
| Q8_0 | 2048×8192 (down_proj) | 2 | 317 | 56.3 | 87% | 79% | 277 | 446 | 62% | 1.61× | 1.41× | 4.29 (3.61–4.54) |
| Q8_0 | 2048×8192 (down_proj) | 4 | 237 | 75.3 | 81% | 53% | 194 | 278 | 69% | 1.43× | 1.17× | 3.60 (3.14–3.65) |
| Q8_0 | 4096×4096 | 1 | 539 | 33.1 | 91% | 100% | 490 | 809 | 61% | 1.65× | 1.50× | 4.24 (3.73–4.48) |
| Q8_0 | 4096×4096 | 2 | 314 | 56.9 | 90% | 86% | 260 | 438 | 65% | 1.68× | 1.39× | 4.17 (3.53–4.52) |
| Q8_0 | 4096×4096 | 4 | 233 | 76.6 | 81% | 58% | 175 | 283 | 66% | 1.61× | 1.21× | 3.64 (3.27–3.66) |
| Q8_0 | 8192×2048 (gate_proj, up_proj) | 1 | 551 | 32.4 | 89% | 100% | 487 | 833 | 59% | 1.71× | 1.51× | 4.42 (3.96–4.58) |
| Q8_0 | 8192×2048 (gate_proj, up_proj) | 2 | 325 | 55.0 | 88% | 85% | 268 | 469 | 61% | 1.75× | 1.44× | 4.35 (3.89–4.57) |
| Q8_0 | 8192×2048 (gate_proj, up_proj) | 4 | 232 | 77.1 | 86% | 59% | 179 | 299 | 66% | 1.67× | 1.29× | 3.56 (3.34–3.67) |
| Q8_0 | 4096×11008 | 1 | 1445 | 33.2 | 91% | 100% | 1422 | 2766 | 48% | 1.94× | 1.91× | 4.43 (3.95–4.51) |
| Q8_0 | 4096×11008 | 2 | 848 | 56.5 | 87% | 85% | 801 | 1511 | 49% | 1.89× | 1.78× | 4.20 (3.57–4.51) |
| Q8_0 | 4096×11008 | 4 | 611 | 78.5 | 83% | 59% | 546 | 911 | 56% | 1.67× | 1.49× | 3.52 (3.06–3.66) |
| Q8_0 | 128256×2048 (lm_head) | 1 | 8037 | 34.8 | 100% | 100% | 8171 | 15579 | 52% | 1.91× | 1.94× | 4.34 (3.95–4.55) |
| Q8_0 | 128256×2048 (lm_head) | 2 | 4710 | 59.4 | 99% | 85% | 4683 | 9754 | 48% | 2.08× | 2.07× | 4.08 (3.53–4.55) |
| Q8_0 | 128256×2048 (lm_head) | 4 | 3234 | 86.4 | 95% | 62% | 3191 | 5819 | 53% | 1.82× | 1.80× | 3.49 (3.31–3.66) |
| Q4_0 | 512×2048 (k_proj, v_proj) | 1 | 29 | 21.0 | 57% | 100% | 22 | 31 | 52% | 1.42× | 1.10× | 4.39 (3.48–4.58) |
| Q4_0 | 512×2048 (k_proj, v_proj) | 2 | 17 | 35.1 | 54% | 83% | 13 | 18 | 50% | 1.43× | 1.08× | 4.32 (3.46–4.52) |
| Q4_0 | 512×2048 (k_proj, v_proj) | 4 | 12 | 48.0 | 50% | 57% | 10 | 12 | 53% | 1.22× | 0.94× | 3.60 (3.23–3.66) |
| Q4_0 | 2048×2048 (q_proj, o_proj) | 1 | 90 | 26.4 | 72% | 100% | 71 | 126 | 52% | 1.77× | 1.39× | 4.13 (3.68–4.56) |
| Q4_0 | 2048×2048 (q_proj, o_proj) | 2 | 56 | 42.4 | 64% | 80% | 38 | 72 | 50% | 1.89× | 1.28× | 4.24 (3.84–4.54) |
| Q4_0 | 2048×2048 (q_proj, o_proj) | 4 | 37 | 64.6 | 70% | 61% | 26 | 43 | 60% | 1.62× | 1.16× | 3.54 (3.18–3.67) |
| Q4_0 | 2048×8192 (down_proj) | 1 | 335 | 28.3 | 78% | 100% | 302 | 593 | 44% | 1.96× | 1.77× | 4.10 (3.54–4.52) |
| Q4_0 | 2048×8192 (down_proj) | 2 | 184 | 51.4 | 84% | 91% | 158 | 311 | 50% | 1.97× | 1.69× | 4.09 (3.72–4.49) |
| Q4_0 | 2048×8192 (down_proj) | 4 | 148 | 63.8 | 69% | 56% | 102 | 187 | 55% | 1.84× | 1.26× | 3.53 (3.14–3.66) |
| Q4_0 | 4096×4096 | 1 | 332 | 28.5 | 78% | 100% | 294 | 553 | 47% | 1.88× | 1.66× | 4.53 (3.72–4.56) |
| Q4_0 | 4096×4096 | 2 | 207 | 45.8 | 71% | 80% | 159 | 298 | 50% | 1.87× | 1.44× | 4.15 (3.79–4.47) |
| Q4_0 | 4096×4096 | 4 | 148 | 64.2 | 68% | 56% | 102 | 187 | 53% | 1.83× | 1.27× | 3.54 (3.01–3.66) |
| Q4_0 | 8192×2048 (gate_proj, up_proj) | 1 | 348 | 27.3 | 76% | 100% | 294 | 555 | 48% | 1.89× | 1.60× | 4.29 (3.87–4.57) |
| Q4_0 | 8192×2048 (gate_proj, up_proj) | 2 | 202 | 47.0 | 75% | 86% | 158 | 318 | 47% | 2.01× | 1.58× | 4.37 (3.80–4.54) |
| Q4_0 | 8192×2048 (gate_proj, up_proj) | 4 | 140 | 67.7 | 71% | 62% | 101 | 188 | 53% | 1.87× | 1.35× | 3.46 (3.07–3.62) |
| Q4_0 | 4096×11008 | 1 | 882 | 28.8 | 84% | 100% | 843 | 2002 | 37% | 2.37× | 2.27× | 4.25 (3.50–4.58) |
| Q4_0 | 4096×11008 | 2 | 492 | 51.7 | 81% | 90% | 466 | 996 | 40% | 2.14× | 2.03× | 3.99 (3.45–4.40) |
| Q4_0 | 4096×11008 | 4 | 338 | 75.3 | 79% | 65% | 264 | 587 | 45% | 2.22× | 1.74× | 3.50 (3.15–3.65) |
| Q4_0 | 128256×2048 (lm_head) | 1 | 5393 | 27.5 | 77% | 100% | 5468 | 9889 | 42% | 1.81× | 1.83× | 4.05 (3.25–4.48) |
| Q4_0 | 128256×2048 (lm_head) | 2 | 2942 | 50.4 | 82% | 92% | 2947 | 6442 | 37% | 2.19× | 2.19× | 4.33 (4.14–4.51) |
| Q4_0 | 128256×2048 (lm_head) | 4 | 1999 | 74.2 | 88% | 67% | 1934 | 3707 | 47% | 1.92× | 1.85× | 3.49 (3.05–3.62) |

**Per-token GEMV time, Llama-3.2-1B** (config.json of meta-llama/Llama-3.2-1B (sha256 bfb5d39c327ee5c3, tie_word_embeddings True); 16 layers × (2 q/o + 2 k/v + 2 gate/up + 1 down) + lm_head, every matrix in the format shown; ours with rotated weights, ggml on the same weights every run; GEMVs only, not a tokens/s figure)

- Q8_0, 1 thread: ours 39.88 ms, ggml 61.33 ms (1.54×)
- Q8_0, 2 threads: ours 23.67 ms, ggml 34.82 ms (1.47×)
- Q8_0, 4 threads: ours 16.95 ms, ggml 21.62 ms (1.28×)
- Q4_0, 1 thread: ours 25.68 ms, ggml 42.16 ms (1.64×)
- Q4_0, 2 threads: ours 14.69 ms, ggml 24.50 ms (1.67×)
- Q4_0, 4 threads: ours 10.43 ms, ggml 14.48 ms (1.39×)

**FP32 / FP16 / BF16 decode GEMV** (x·Wᵀ, W [N, K]), same roof and scaling; PyTorch: `F.linear` at `torch.set_num_threads(N)`, timed alone after ours (two OpenMP runtimes would spin against each other), FP32 and BF16.

| weights | N×K | threads | simd-ml-kernels µs | GB/s | % of read roof | scaling | PyTorch GB/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 2048×8192 | 1 | 2147 | 31.3 | 92% | 100% | 32.2 | 0.97× | 4.00 (3.14–4.45) |
| FP32 | 2048×8192 | 2 | 1126 | 59.6 | 93% | 95% | 56.0 | 1.07× | 4.14 (3.61–4.42) |
| FP32 | 2048×8192 | 4 | 822 | 81.7 | 85% | 65% | 81.1 | 1.01× | 3.48 (3.12–3.64) |
| FP32 | 4096×4096 | 1 | 1986 | 33.8 | 96% | 100% | 33.4 | 1.01× | 4.42 (3.66–4.54) |
| FP32 | 4096×4096 | 2 | 1153 | 58.2 | 89% | 86% | 58.4 | 1.00× | 4.28 (3.83–4.54) |
| FP32 | 4096×4096 | 4 | 816 | 82.3 | 91% | 61% | 82.6 | 1.00× | 3.59 (3.24–3.67) |
| FP32 | 4096×11008 | 1 | 5153 | 35.0 | 101% | 100% | 33.8 | 1.03× | 4.29 (3.73–4.59) |
| FP32 | 4096×11008 | 2 | 3060 | 59.0 | 93% | 84% | 58.3 | 1.01× | 4.18 (3.50–4.55) |
| FP32 | 4096×11008 | 4 | 2106 | 85.7 | 89% | 61% | 84.6 | 1.01× | 3.57 (3.17–3.66) |
| FP16 | 2048×8192 | 1 | 1110 | 30.2 | 83% | 100% | — | — | 4.28 (3.97–4.51) |
| FP16 | 2048×8192 | 2 | 657 | 51.1 | 80% | 84% | — | — | 4.17 (3.67–4.47) |
| FP16 | 2048×8192 | 4 | 457 | 73.4 | 77% | 61% | — | — | 3.58 (3.16–3.66) |
| FP16 | 4096×4096 | 1 | 1067 | 31.5 | 87% | 100% | — | — | 4.50 (3.94–4.58) |
| FP16 | 4096×4096 | 2 | 647 | 51.9 | 85% | 82% | — | — | 3.79 (3.37–4.53) |
| FP16 | 4096×4096 | 4 | 445 | 75.5 | 83% | 60% | — | — | 3.50 (3.24–3.65) |
| FP16 | 4096×11008 | 1 | 2617 | 34.5 | 96% | 100% | — | — | 4.26 (3.71–4.55) |
| FP16 | 4096×11008 | 2 | 1588 | 56.8 | 85% | 82% | — | — | 4.25 (3.78–4.47) |
| FP16 | 4096×11008 | 4 | 1113 | 81.1 | 84% | 59% | — | — | 3.59 (3.01–3.66) |
| BF16 | 2048×8192 | 1 | 1118 | 30.0 | 86% | 100% | 18.8 | 1.60× | 4.46 (3.95–4.88) |
| BF16 | 2048×8192 | 2 | 609 | 55.2 | 80% | 92% | 33.5 | 1.65× | 4.53 (4.08–4.82) |
| BF16 | 2048×8192 | 4 | 424 | 79.2 | 75% | 66% | 53.3 | 1.48× | 3.81 (3.18–3.94) |
| BF16 | 4096×4096 | 1 | 1096 | 30.6 | 85% | 100% | 17.4 | 1.76× | 4.14 (3.93–4.53) |
| BF16 | 4096×4096 | 2 | 649 | 51.7 | 81% | 84% | 31.8 | 1.62× | 4.12 (3.56–4.49) |
| BF16 | 4096×4096 | 4 | 446 | 75.3 | 79% | 61% | 49.8 | 1.51× | 3.63 (3.24–3.66) |
| BF16 | 4096×11008 | 1 | 2559 | 35.3 | 98% | 100% | 18.0 | 1.96× | 4.53 (3.78–4.56) |
| BF16 | 4096×11008 | 2 | 1691 | 53.4 | 91% | 76% | 33.6 | 1.59× | 3.78 (3.33–4.52) |
| BF16 | 4096×11008 | 4 | 1112 | 81.1 | 83% | 58% | 53.1 | 1.53× | 3.54 (3.22–3.65) |
<!-- END GENERATED: threads-summary -->

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
| Python | 3.9–3.13 (for Python extension; wheels for these versions) |
| pybind11 | ≥ 2.11 (for Python extension) |
| AVX2 + FMA + F16C | At run time, for the AVX2 kernels (scalar kernels otherwise) |

Optional: `libopenblas-dev` for `./bench_stat` baseline comparisons (`-DBENCH_OPENBLAS=ON`).

---

## Repository Layout

The map of the source tree, the build/test/benchmark commands and the checklist
for adding a kernel are in
[AGENTS.md](https://github.com/Mattral/simd-ml-kernels/blob/main/AGENTS.md);
design notes are in
[docs/DESIGN.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/DESIGN.md).

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
[docs/DESIGN.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/DESIGN.md#8-ci) §8.

---

## Contributing

Issues, pull requests and benchmark results from other hardware are welcome:
run `python -m simd_ml_kernels bench --json` with the protocol in
[docs/BENCHMARKS.md](https://github.com/Mattral/simd-ml-kernels/blob/main/docs/BENCHMARKS.md)
and open a PR with the JSON. Start with
[AGENTS.md](https://github.com/Mattral/simd-ml-kernels/blob/main/AGENTS.md) and
[CONTRIBUTING.md](https://github.com/Mattral/simd-ml-kernels/blob/main/CONTRIBUTING.md).

---

## Naming

One name everywhere: repository and PyPI distribution `simd-ml-kernels`,
import `simd_ml_kernels`. The project was previously called IntrinsicML (the
preprint title), the repository `SIMD-Microkernels-for-ML-Workloads` (GitHub
redirects old links), and the import `simd_kernels`, which still works but
emits a `DeprecationWarning` and will be removed in 0.5.0. See
[CHANGELOG.md](https://github.com/Mattral/simd-ml-kernels/blob/main/CHANGELOG.md).

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

Apache 2.0 — see
[LICENSE](https://github.com/Mattral/simd-ml-kernels/blob/main/LICENSE).
