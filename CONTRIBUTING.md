# Contributing to simd-ml-kernels

Thank you for your interest in contributing. simd-ml-kernels values transparency, correctness, and rigour above all else. A pull request that is correct and well-documented is always preferred over one that is fast but opaque.

---

## Quick start

```bash
git clone https://github.com/Mattral/simd-ml-kernels.git
cd simd-ml-kernels

# C++ build and tests
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Python extension and tests
pip install -e . --no-build-isolation
pytest tests/test_precision.py tests/test_bindings_edge_cases.py -v
```

See `docs/setup.md` for full prerequisites and optional flags.

---

## Where to contribute

### High-value areas (aligned with ROADMAP.md)

| Area | Status | Entry point |
|---|---|---|
| Skip panel packing for small N (packed kernel is slower than scalar at N=64) | 🔴 Planned v0.9 | `src/kernels/gemm/avx2_gemm_packed.cpp` |
| BF16/FP16 GEMM | 🔴 Planned v0.9 | new file under `src/kernels/gemm/` |
| Auto-tuned tile sizes (MC/KC/NC re-derived for Sapphire Rapids / Zen 4) | 🔴 Planned v0.9 | `src/kernels/gemm/avx2_gemm_packed.hpp` |
| Benchmark results on new hardware | 🟡 Always welcome | `docs/BENCHMARKS.md` |
| Bug reports with repro steps | 🟡 Always welcome | GitHub Issues |

Before starting significant work, open an issue first to avoid duplicating effort.

---

## Coding standards

### C++ (C++17)

**Naming**
```cpp
// Types and classes: PascalCase
struct KernelRegistry { ... };

// Functions and variables: snake_case
void sgemm_packed(int M, int N, ...);
float max_rel_err = 0.0f;

// Constants and compile-time values: SCREAMING_SNAKE or constexpr
static constexpr int MC = 64;
static constexpr int SIMD_WIDTH = 8;

// Private/static helpers: static, snake_case, in .cpp file (not header)
static void pack_b_panel(...);
```

**Comments — the most important rule**
Every non-trivial design decision must be documented where it is made. This is the core value of simd-ml-kernels. Comments explain *why*, not *what*:

```cpp
// GOOD: explains the design decision
// Unroll ×4 to expose 32 FMAs to the scheduler simultaneously,
// hiding the 4-cycle FMA latency without needing a separate pipelining stage.
for (; k <= kc - 4; k += 4) { ... }

// BAD: just restates the code
// Loop over k, unrolled by 4
for (; k <= kc - 4; k += 4) { ... }
```

For any new kernel, include at the top of the file:
1. Mathematical formula being computed
2. SIMD strategy (which intrinsics, why this width)
3. Max absolute error vs reference (for approximation kernels)
4. Cache/register pressure analysis
5. Known limitations

**Correctness before performance**

Every optimised path must have a scalar reference in the same file that documents the mathematical operation. Tests compare against the reference.

```cpp
// REQUIRED for any new kernel:
void my_kernel_avx2(const float* in, float* out, int n);   // fast path
void my_kernel_scalar(const float* in, float* out, int n); // oracle
```

**Memory**

All buffers must use `make_aligned_array<T>()` from `cache_alloc.hpp` (64-byte aligned). Never `new float[n]` or `std::vector<float>` for buffers passed to SIMD kernels.

**Error handling**

Kernels do not throw. Input validation happens at the Python binding layer (`pybind_entry.cpp`). Internal preconditions use `assert()` (stripped by `-DNDEBUG` in production builds).

**Warnings**

The project builds with `-Wall -Wextra -Wshadow`. Your code must produce zero warnings. Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2>&1 | grep warning
```

### Python

- **Python ≥ 3.9**, type hints on all public functions
- All public functions must be in `python/simd_ml_kernels/_core.pyi` (type stubs)
- Tests use `pytest`. New kernels need tests in `tests/test_precision.py`
- Test tolerance rationale must be commented: why is this tolerance correct?

---

## Adding a new kernel

Follow this checklist exactly. Skipping steps is the primary source of bugs in this codebase.

**Step 1: Scalar reference**
Write the mathematically correct scalar implementation first. No SIMD yet. This is your correctness oracle.

```cpp
// src/kernels/activations/my_kernel_avx2.cpp
static void my_kernel_scalar(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        // exact formula, no approximations
        out[i] = std::some_function(in[i]);
    }
}
```

**Step 2: Measure scalar accuracy**
Compute max absolute error of your scalar implementation against a double-precision reference. Document the result.

**Step 3: Declare in `activations.hpp`**
Add a declaration with a comment stating the algorithm and error bound.

**Step 4: AVX2 implementation**
Use `simd_math.hpp` for `fast_exp_avx2` and `tanh_avx2` rather than re-implementing them. Document every non-obvious intrinsic choice.

**Step 5: Scalar tail**
All AVX2 loops must have a scalar tail for `n % 8` remaining elements.

**Step 6: C++ correctness test**
Add cases to `tests/test_activations.cpp` following the existing pattern. Test: edge sizes (1, 7, 8, 9), full AVX2 sizes (64, 1024), wide sweeps over the input range.

**Step 7: Python binding**
Add to `pybind_entry.cpp`, `python/simd_ml_kernels/_core.pyi`, and document in `docs/api/python_api.md`.

**Step 8: Python precision test**
Add a `TestMyKernel` class to `tests/test_precision.py`. Include a `@pytest.mark.skipif(not HAS_TORCH, ...)` cross-check against PyTorch if applicable.

**Never use a naive relative-error metric** — `|actual - ref| / (|ref| + eps)`
with a tiny `eps` (e.g. `1e-7`) is numerically unstable for any function that
crosses zero (GeLU, SiLU, LayerNorm, tanh-based activations all qualify).
When `ref` is legitimately near zero, a few ULPs of disagreement between two
independent implementations get amplified into an arbitrarily large
"relative error" that has nothing to do with kernel correctness. This is not
theoretical: `test_gelu_vs_pytorch_tanh_approx` used exactly this pattern and
failed on **68% of random seeds** when stress-tested, despite the underlying
kernel being correct to within a few ULPs of PyTorch's own implementation.
Always use the shared `assert_close(actual, ref, rtol, atol, label=...)`
helper in `tests/test_precision.py`, which implements the combined
`|diff| <= atol + rtol*|ref|` criterion (the same one `np.allclose` uses).
Pick `atol` based on the kernel's documented absolute error bound near zero
(typically `1e-5` to `1e-6` for the tanh-based activations in this project).

**Step 9: Run all tests**
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
pytest tests/test_precision.py tests/test_bindings_edge_cases.py -v
```

All must pass with zero warnings.

**Step 10: Update ROADMAP.md**
Mark the item ✅ in ROADMAP.md.

---

## Adding a GEMM tile variant

GEMM tile variants (`MR`, `NR`, `MC`, `KC`, `NC`) require special care:

1. Derive the new constants from first principles (show your cache-level derivation, see `docs/design/cache_hierarchy.md`)
2. Add a `#define VARIANT_NAME` compile-time guard
3. Test with the *existing* `test_gemm_packed.cpp` test suite — all sizes must pass, including non-power-of-two
4. Benchmark with `bench_stat --reps 50` on a pinned core and document the result in `docs/BENCHMARKS.md`
5. Do not change the default tile sizes without a clear documented performance improvement on representative hardware

---

## Pull request process

1. **One logical change per PR.** A PR that fixes a bug and adds a feature is two PRs.
2. **Description template:**
   - What does this change?
   - Why is this the right approach? (Link to design doc or issue)
   - What correctness guarantees does it maintain?
   - What are the measured performance effects (if any)?
3. **CI must pass.** Every workflow (ci.yml, guardrail.yml) must be green. The weekly bench.yml is not required to be green for a PR (it only runs on schedule), but the change must not be expected to cause a regression.
4. **No force-pushes** to main after merging.

---

## Benchmark contributions

If you run benchmarks on hardware not yet represented in `docs/BENCHMARKS.md`, contributions are especially valuable. Please include:

- Exact CPU model (`lscpu | grep "Model name"`)
- Cache sizes (`lscpu | grep -E "L1d|L2|L3"`)
- Compiler version (`g++ --version`)
- Whether CPU frequency was locked (`cpupower frequency-info`)
- Full `bench_stat --reps 50` output as JSON (commit to `benchmarks/results/`)

---

## Code of conduct

Be direct and technical. Critique code, not people. Assume good faith. Performance engineering is hard — a wrong first approach is the normal path to a correct one.

---

## License

By contributing, you agree that your contribution will be licensed under the Apache 2.0 License.
