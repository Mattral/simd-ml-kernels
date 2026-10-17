# Security Considerations

---

## Threat Model

simd-ml-kernels is a CPU microkernel library intended for use in research,
benchmarking, and production ML inference pipelines. It is **not** a
network-facing service or a security boundary. The threat model covers:

- **Memory safety** in native C++ kernel code
- **Input validation** at the Python binding layer
- **Dependency hygiene** for the build system and runtime dependencies
- **Supply-chain integrity** for CI/CD artifacts

---

## Native Code Safety

### What the kernels do

All kernels operate on contiguous float32 arrays via raw pointers. The inner
loops access memory in a predictable strided pattern bounded by the caller-supplied
dimensions M, N, K, n.

### Validation layer

**All dimension and type validation happens in `pybind_entry.cpp`** before any
pointer is passed to a kernel:

```cpp
// Validated before calling any C++ kernel:
// 1. dtype == float32 (not float64, int, etc.)
// 2. ndim matches expected (e.g., sgemm requires 2-D inputs)
// 3. C-contiguous memory layout
// 4. Writeable flag for in-place operations
// 5. Shape compatibility (K-dimension match for GEMM)
// 6. Gamma/beta size matching last dimension for layer_norm
```

A malformed Python call raises a `RuntimeError` with a descriptive message
before touching any native memory.

### Size bounds and overflow

All tile-loop bounds are derived from caller-supplied dimensions (M, N, K ≥ 0).
The guards `if (M <= 0 || N <= 0 || K <= 0) return;` in both GEMM entry points
prevent zero or negative sizes from reaching the inner loops.

Large inputs (e.g., M=N=K=1,000,000) would require 4 TB of FP32 storage —
the OS will refuse the allocation and a `std::bad_alloc` exception propagates
cleanly to Python as a `MemoryError`.

### Recommended mitigations for contributors

```bash
# Build with AddressSanitizer + UndefinedBehaviorSanitizer:
cmake -S . -B build_asan -DSIMD_ML_SANITIZE=ON -DCMAKE_CXX_COMPILER=clang++-15
cmake --build build_asan
ctest --test-dir build_asan

# Run the static analysis workflow locally:
clang-tidy-15 -p build_asan src/kernels/*.cpp src/kernels/activations/*.cpp \
    src/kernels/gemm/*.cpp
```

---

## Python Binding Safety

### GIL release

All kernel calls release the GIL (`py::gil_scoped_release`). This enables
concurrent Python threads but means kernel execution is non-interruptible from
Python during the call. This is safe because:
- Kernels are pure compute (no Python API calls during execution)
- Inputs are validated before GIL release
- No shared mutable state between concurrent kernel calls

### Buffer aliasing

`gelu_forward_avx2` explicitly supports in-place operation (`input == output`).
Other kernels use separate input/output pointers; aliased in/out pointers
would cause undefined behaviour and are guarded by the `__restrict__` qualifier
on the C++ side and the `writeable` flag check on the Python side.

---

## Dependency Hygiene

**Runtime dependencies** (minimal by design):

| Dependency | Version | Notes |
|---|---|---|
| `numpy` | ≥ 1.24 | Runtime; NumPy's own security policy applies |
| `pybind11` | ≥ 2.11 | Build-time only; not bundled in the wheel |

**No internet access at runtime.** The library is self-contained once installed.

**For contributors**: pin build toolchains in CI and the Dockerfile to avoid
silent upstream changes. The weekly `guardrail.yml` workflow runs `pip-audit`
to check for known vulnerabilities in Python dependencies.

---

## Supply-Chain Integrity

- All CI/CD workflows are defined in `.github/workflows/` and reviewed on every PR
- `FetchContent` in CMakeLists.txt downloads doctest at a pinned tag (`v2.4.12`) via HTTPS
- The Docker image pins GCC, Python, CMake, pybind11, numpy, and pytest versions
- Release artifacts are built from clean CI runs on trusted GitHub-hosted runners

**Not yet implemented** (planned for v1.0):
- Signed releases (GPG or GitHub Attestations)
- SBOM (Software Bill of Materials) generation
- Reproducible builds (currently `--build-timestamp` embeds wall-clock time)
