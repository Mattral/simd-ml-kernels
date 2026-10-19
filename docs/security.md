# Security Considerations

---

## Threat Model

simd-ml-kernels is a CPU kernel library for research, benchmarking and ML
inference code. It is **not** a network-facing service or a security
boundary. The threat model covers:

- **Memory safety** in native C++ kernel code
- **Input validation** at the Python binding layer
- **Dependency hygiene** for the build system and runtime dependencies
- **Supply-chain integrity** for CI/CD artifacts

---

## Native Code Safety

### What the kernels do

The kernels operate on contiguous arrays (float32; float16, BF16 or
uint8/int8 inputs for the low-precision GEMMs) through raw pointers. Every
loop is bounded by the caller-supplied dimensions M, N, K or n; remainders use
masked loads and stores or a copied tail, so nothing outside the arrays is
read or written (docs/DESIGN.md §2.1, §2.7).

### Validation layer

**All dimension and type validation happens in `pybind_entry.cpp`** before any
pointer is passed to a kernel:

```cpp
// Validated before calling any C++ kernel:
// 1. dtype (float32; float16 / BF16 / uint8 / int8 for the low-precision GEMMs)
// 2. ndim matches expected (e.g., sgemm requires 2-D inputs)
// 3. C-contiguous memory layout
// 4. Writeable flag for in-place operations and caller-supplied outputs
// 5. Shape compatibility (K-dimension match for GEMM, b_scales length for INT8)
// 6. Gamma/beta size matching last dimension for layer_norm
```

A malformed Python call raises `simd_ml_kernels.DTypeError` (a `TypeError`)
or `simd_ml_kernels.ArgumentError` (a `ValueError`) with a descriptive message
before touching any native memory; both also subclass `RuntimeError`
(docs/api/python_api.md, "Error Handling").

### Size bounds and overflow

All loop bounds are derived from caller-supplied dimensions. The GEMM entry
points return immediately for `M <= 0` or `N <= 0`; `K == 0` is valid (BLAS
semantics) and only applies `beta` to C.

The kernels allocate nothing except the GEMM packing buffers, which are kept
per thread and grown on demand. If growing them fails, `std::bad_alloc`
propagates to Python as `MemoryError`.

### Recommended mitigations for contributors

```bash
# Build with AddressSanitizer + UndefinedBehaviorSanitizer:
cmake -S . -B build_asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DSIMD_ML_SANITIZE=ON -DCMAKE_CXX_COMPILER=clang++-15
cmake --build build_asan
ctest --test-dir build_asan --output-on-failure
# On WSL2, ASan binaries can crash before main (ASLR): setarch -R ctest ...

# Run the static analysis workflow locally (needs a compilation database):
cmake -S . -B build_tidy -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
clang-tidy-15 -p build_tidy $(find src/kernels src/dispatch -name "*.cpp")
```

---

## Python Binding Safety

### GIL release

All kernel calls release the GIL (`py::gil_scoped_release`). This enables
concurrent Python threads but means kernel execution is non-interruptible from
Python during the call. This is safe because:
- Kernels are pure compute (no Python API calls during execution)
- Inputs are validated before GIL release
- The only state a kernel keeps between calls is its thread's own packing
  buffers (`thread_local`), so concurrent calls from different threads do not
  share mutable state (`tests/test_threads.cpp` runs concurrent GEMMs from
  the threads of an OpenMP region in OpenMP builds)

### Buffer aliasing

`gelu_inplace` (both GeLU variants) is written to work with input and output
being the same buffer. The other functions always write into a different
array from their inputs: either one they allocate, or a caller-supplied `C`.
Passing a `C` that overlaps `A` or `B` is not checked and gives undefined
results.

---

## Dependency Hygiene

**Runtime dependencies** (minimal by design):

| Dependency | Version | Notes |
|---|---|---|
| `numpy` | ≥ 1.24 | Runtime; NumPy's own security policy applies |
| `pybind11` | ≥ 2.11 | Build-time only; not bundled in the wheel |

**No internet access at runtime.** The library is self-contained once installed.

**For contributors**: pin build toolchains in CI and the Dockerfile to avoid
silent upstream changes. The `guardrail.yml` workflow (every push and pull
request, and weekly) runs `pip-audit` on the Python dependencies; it reports
and does not block.

---

## Supply-Chain Integrity

- All CI/CD workflows are defined in `.github/workflows/` and reviewed on every PR
- The CMake build downloads nothing; the C++ tests have no external test framework
- The Docker image pins GCC 12 on Ubuntu 22.04 and minimum versions of CMake's Python dependencies
- Wheels and the sdist are built and selftested by `wheels.yml` on GitHub-hosted
  runners and uploaded only by the manual `publish.yml`, from `main`, with
  PyPI Trusted Publishing (OIDC; no API token or repository secret). The upload
  action also generates PEP 740 attestations for the files. 0.3.0rc1 was
  published this way to TestPyPI; nothing is on PyPI yet.

**Not yet implemented**:
- SBOM (Software Bill of Materials) generation
- Reproducible builds (`build_info()["build_timestamp"]` embeds the compile
  time through `__DATE__` / `__TIME__`)
