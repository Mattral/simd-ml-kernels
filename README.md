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
- Q4_0 decode GEMV (laptop, `42f64c7d76b3`, 1×4096×11008): 79% of the read
  roof, 2.16× ggml's Q4_0 MUL_MAT (llama.cpp v0.7.0, one thread), 1.67× our
  Q8_0 GEMV.
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
- Single-threaded by default: OpenMP GEMM is a build option without a
  published scaling table; the GEMV and the INT8 GEMM have no threaded path.
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

Source: Intel(R) Core(TM) Ultra 5 226V, WSL2 vCPU 2 (P-core by measured clock, see protocol); Python 3.12.3, GNU 13.3.0, commit `42f64c7d76b3`; baselines: NumPy 2.5.4 with scipy-openblas 0.3.34.106.0, PyTorch 2.14.1+cpu; 30 interleaved reps after 5 warm-up; 2026-10-20T02:14:20+00:00; [`benchmarks/results/lnl-226v-wsl_2026-10-20_42f64c7d76b3.json`](https://github.com/Mattral/simd-ml-kernels/blob/main/benchmarks/results/lnl-226v-wsl_2026-10-20_42f64c7d76b3.json).

**FP32 GEMM, square, single thread**

| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS | core GHz median (min–max) |
|---:|---:|---:|---:|---:|
| 128 | 92.7 | 99.5 | 93% | 4.10 (3.70–4.42) |
| 256 | 114.3 | 116.3 | 98% | 4.09 (3.76–4.37) |
| 1024 | 109.6 | 112.8 | 97% | 3.74 (3.63–4.32) |
Not shown: N = 512 (median 3.98, min 3.17 GHz, 3 attempts), N = 2048 (median 3.98, min 3.58 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**FP16 / BF16-input GEMM, FP32 accumulation and output, single thread** (PyTorch: `torch.matmul` on float16 / bfloat16 tensors, which returns float16 / bfloat16)

| input | M×N×K | simd-ml-kernels GFLOP/s | our FP32 GFLOP/s | % of FP32 | PyTorch GFLOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | 128×128×128 | 95.1 | 96.0 | 99% | 61.1 | 1.56× | 4.20 (3.68–4.35) |
| FP16 | 256×256×256 | 109.3 | 116.1 | 94% | 76.9 | 1.42× | 4.09 (3.84–4.24) |
| FP16 | 512×512×512 | 105.8 | 107.3 | 99% | 78.4 | 1.35× | 3.95 (3.60–4.37) |
| FP16 | 1024×1024×1024 | 99.2 | 100.5 | 99% | 69.7 | 1.42× | 3.91 (3.63–4.19) |
| FP16 | 2048×2048×2048 | 107.2 | 106.8 | 100% | 74.6 | 1.44× | 3.96 (3.63–4.45) |
| FP16 | 1×4096×4096 | 5.2 | 4.1 | 126% | 9.9 | 0.53× | 4.15 (3.68–4.39) |
| BF16 | 128×128×128 | 92.2 | 90.0 | 102% | 72.8 | 1.27× | 4.05 (3.79–4.29) |
| BF16 | 256×256×256 | 106.7 | 109.9 | 97% | 101.9 | 1.05× | 4.19 (3.77–4.41) |
| BF16 | 512×512×512 | 110.7 | 108.2 | 102% | 111.9 | 0.99× | 4.17 (3.69–4.37) |
Not shown: BF16 1024×1024×1024 (median 3.95, min 3.29 GHz, 3 attempts), BF16 2048×2048×2048 (median 3.77, min 3.20 GHz, 3 attempts), BF16 1×4096×4096 (median 4.05, min 3.47 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**INT8 GEMM (uint8 × per-channel int8, FP32 output), x·Wᵀ, single thread**, path `avx_vnni` (PyTorch: `torch.ops.quantized.linear`, prepacked weights, which returns requantized uint8)

| M×N×K | simd-ml-kernels GOP/s | × our FP32 | FP32-conversion path GOP/s | × our FP32 | PyTorch GOP/s | ours / PyTorch | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×512×512 | 393.4 | 3.67× | 109.9 | 1.03× | 208.4 | 1.89× | 3.89 (3.74–4.40) |
Not shown: 128×128×128 (median 4.29, min 3.36 GHz, 3 attempts), 256×256×256 (median 3.91, min 3.49 GHz, 3 attempts), 1024×1024×1024 (median 4.05, min 3.37 GHz, 3 attempts), 2048×2048×2048 (median 3.66, min 3.31 GHz, 3 attempts), 1×4096×4096 (median 4.15, min 3.35 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**Decode GEMV y = x·Wᵀ (one row x, W [N, K] as torch.nn.Linear stores it), FP32 output, single thread**; read roof: the read stream timed in each case, interleaved with it; STREAM triad of the run: single-core read stream 34.5 GB/s, STREAM triad 27.9 GB/s (all-core: read not measured, triad not measured). GB/s counts the weights, x and y. (PyTorch: `F.linear` on the same weights in their dtype, which returns that dtype; OpenBLAS: NumPy's matrix-vector product, FP32 only)

| weights | M×N×K | simd-ml-kernels µs | GB/s | % of read roof | % of triad | PyTorch GB/s | ours / PyTorch | OpenBLAS sgemv GB/s | ours / OpenBLAS | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 1×2048×8192 | 2127 | 31.6 | 98% | 113% | 30.1 | 1.05× | 27.6 | 1.15× | 4.04 (3.76–4.42) |
| FP32 | 1×4096×4096 | 2216 | 30.3 | 91% | 109% | 30.3 | 1.00× | 29.8 | 1.02× | 4.33 (4.11–4.41) |
| FP16 | 1×2048×8192 | 1227 | 27.4 | 88% | 98% | 17.3 | 1.58× | — | — | 3.95 (3.72–4.35) |
| BF16 | 1×2048×8192 | 1167 | 28.8 | 87% | 103% | 16.1 | 1.79× | — | — | 3.80 (3.62–4.36) |
| BF16 | 1×4096×4096 | 1236 | 27.2 | 86% | 98% | 15.5 | 1.75× | — | — | 4.09 (3.70–4.39) |
Not shown: FP32 1×4096×11008 (median 3.98, min 3.58 GHz, 3 attempts), FP16 1×4096×4096 (median 4.23, min 3.45 GHz, 3 attempts), FP16 1×4096×11008 (median 3.76, min 3.52 GHz, 3 attempts), BF16 1×4096×11008 (median 4.02, min 3.22 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**Q8_0 decode GEMV** (`gemv_q8_0`: ggml block_q8_0 weights, 34 bytes per 32, x quantized to Q8 inside), path `avx_vnni`, single thread; GB/s counts the weights, x and y; × columns: time of that GEMV / ours, in the same interleaved case. PyTorch: FP32 `F.linear` on the dequantized weights, not a quantized kernel. ggml: llama.cpp v0.7.0 (`620d222`), `test-backend-ops perf` MUL_MAT q8_0 × f32, one thread (OMP_THREAD_LIMIT=1, GGML_OPENMP=ON, GGML_NATIVE=ON), median of 3 runs of its own mean over ≥ 1 s on the same weights (not interleaved).

| M×N×K | simd-ml-kernels µs | GB/s | % of read roof | × our FP32 GEMV | × our BF16 GEMV | PyTorch FP32 µs | ours / PyTorch | ggml µs (own timing) | ggml GB/s | ggml % of roof | ours / ggml | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×2048×8192 | 673 | 26.5 | 82% | 3.19× | 1.73× | 2220 | 3.30× | 987 | 18.1 | 56% | 1.47× | 4.00 (3.81–4.34) |
| 1×4096×4096 | 609 | 29.3 | 91% | 3.55× | 1.99× | 2319 | 3.81× | 1012 | 17.6 | 55% | 1.66× | 4.23 (3.75–4.40) |
Not shown: 1×4096×11008 (median 3.60, min 3.21 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**Q4_0 decode GEMV** (`gemv_q4_0`: ggml block_q4_0 weights, 18 bytes per 32, x quantized to Q8 inside), path `avx_vnni`, single thread; GB/s counts the weights, x and y; × columns: time of that GEMV / ours, in the same interleaved case (our Q8_0 GEMV on the Q8_0 blocks of the same W). PyTorch: FP32 `F.linear` on the dequantized weights, not a quantized kernel. ggml: llama.cpp v0.7.0 (`620d222`), `test-backend-ops perf` MUL_MAT q4_0 × f32, one thread (OMP_THREAD_LIMIT=1, GGML_OPENMP=ON, GGML_NATIVE=ON), median of 3 runs of its own mean over ≥ 1 s on the same weights (not interleaved).

| M×N×K | simd-ml-kernels µs | GB/s | % of read roof | × our Q8_0 GEMV | × our FP32 GEMV | PyTorch FP32 µs | ours / PyTorch | ggml µs (own timing) | ggml GB/s | ggml % of roof | ours / ggml | core GHz median (min–max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×4096×11008 | 1028 | 24.7 | 79% | 1.67× | 6.30× | 5951 | 5.79× | 2221 | 11.4 | 37% | 2.16× | 4.12 (3.73–4.42) |
Not shown: 1×2048×8192 (median 4.03, min 3.56 GHz, 3 attempts), 1×4096×4096 (median 3.71, min 3.35 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

**Activations and normalization, out-of-place, single thread**

| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch | core GHz median (min–max) |
|---|---|---:|---:|---:|---:|---:|
| `gelu_exact` | 1048576 | 2.591 | 0.643 | 1.510 | 1.72× | 4.03 (3.60–4.43) |
| `gelu_tanh` | 1048576 | 1.442 | 0.368 | 0.499 | 2.89× | 3.92 (3.75–4.42) |
| `relu` | 1048576 | 4.712 | 1.139 | 4.409 | 1.07× | 4.14 (3.79–4.40) |
| `softmax` | 1024×1024 | 1.671 | 0.415 | 1.334 | 1.25× | 4.02 (3.61–4.30) |
| `layer_norm` | 1024×1024 | 3.965 | 0.973 | 4.043 | 0.98× | 4.08 (3.77–4.36) |
Not shown: `silu` 1048576 (median 4.10, min 3.59 GHz, 3 attempts). A clock sample stayed below the 3.6 GHz P-core threshold (`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.

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
