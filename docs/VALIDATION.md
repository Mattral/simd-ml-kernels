# Pre-release validation: 0.4.0

What ran on a real CPU before the 0.4.0 release, and what did not. Each row
is a command and its exit status or test count. These are correctness checks,
not benchmarks: no timing is recorded here.

## Machine

| | |
|---|---|
| Date | 2026-11-21 |
| CPU | Intel Core Ultra 5 226V (Lunar Lake), 8 logical CPUs |
| SIMD features (CPUID) | AVX2, FMA, F16C, AVX-VNNI. No AVX-512F/DQ/BW/VL, no AVX512_VNNI, no AVX512_BF16 |
| OS | Windows, WSL2 (kernel 6.18.58.1-microsoft-standard-WSL2), Ubuntu 24.04.5 LTS |
| Toolchain | Python 3.12.3, GCC 13.3.0 (pip build), g++-12 12.4.0 (ctest build), CMake 3.28.3 |
| Libraries | NumPy 2.5.4, PyTorch 2.14.1+cpu (the PyTorch cross-checks in pytest ran) |
| Commit | `9b3a8088e77d` (`main`), and `6459947f4570` for the OpenMP build (rows 11–12; built from that commit's working tree with uncommitted doc edits only, so `git_commit` reads `-dirty`); the commits between them change only docs, charts and CI checks |
| Build | `pip install --no-build-isolation .` (no OpenMP); the same with `SKBUILD_CMAKE_DEFINE=SIMD_ML_OPENMP=ON` (rows 11–12); a Release CMake build with g++-12 |

## Commands and results

| # | Command | Result |
|---|---|---|
| 1 | `python -m simd_ml_kernels info` | exit 0. `isa_compiled` scalar avx2 avx512, `runtime_isa` avx2, `u8s8_path` avx_vnni, `q8_0_path` avx_vnni, `avx512f_dq` False, `avx512_bf16` False, `openmp` False |
| 2 | `python -m simd_ml_kernels selftest` | passed: 16 checks on each of the scalar and avx2 tables (active: avx2) |
| 3 | `SIMD_ML_ISA=scalar python -m simd_ml_kernels selftest` | passed (active: scalar; `u8s8_path` and `q8_0_path` scalar) |
| 4 | `SIMD_ML_ISA=avx2 python -m simd_ml_kernels selftest` | passed (active: avx2) |
| 5 | `SIMD_ML_ISA=avx512 python -m simd_ml_kernels selftest` | passed. The cap falls back to avx2 because this CPU cannot run the avx512 table |
| 6 | `sk.sgemm(A, B, isa=...)` with `scalar`, `avx2`, `avx512` | scalar and avx2 ran. `avx512` raised `RuntimeError` (this CPU cannot run it), as documented |
| 7 | `python -m pytest tests` | 1000 passed, 86 skipped. Every module ran once on the scalar and once on the avx2 table. Skips: 74 need an OpenMP build, 8 need the Llama weights (`SIMD_ML_LLAMA_DIR`), 4 need ggml (`SIMD_ML_GGML_LIB`) |
| 8 | `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-12`, then `cmake --build build` and `ctest --test-dir build` | 2/2 passed (`simd_tests`, `isa_units`). The one compiler warning is in pybind11's headers (`-Wstringop-overread`), not in this project's sources |
| 9 | `sde64 -skx -- python -m simd_ml_kernels selftest` | **not run**: Intel SDE is not installed in this WSL2 environment |
| 10 | `sde64 -spr -- python -m simd_ml_kernels selftest` | **not run**: Intel SDE is not installed in this WSL2 environment |
| 11 | OpenMP build: `python -m simd_ml_kernels selftest` | passed (`openmp` True, 8 threads by default) |
| 12 | OpenMP build: `python -m pytest tests` | 1072 passed, 14 skipped. The 74 OpenMP tests ran and passed. Skips: 2 need a build without OpenMP, 8 need the Llama weights, 4 need ggml |
| 13 | `SIMD_ML_LLAMA_DIR=~/code/models/Llama-3.2-1B python -m pytest tests/test_llama.py tests/test_llama_gguf.py` (default build, transformers 5.20.2, PyTorch 2.14.1+cpu) | 7 passed, 1 skipped. Passed: tokenizer and RoPE frequencies match transformers, logits of the first positions within tolerance, greedy decode token-identical, GGUF Q8_0/Q4_0 blocks match our quantizers, F32 GGUF greedy decode matches transformers. Skipped: the perplexity-text tokenization check, which needs llama.cpp's `llama-tokenize` (`SIMD_ML_GGML_TEST_BACKEND_OPS` not set) |
| 14 | `python -m build`, then `twine check --strict dist/*` (twine 7.0.0, build 1.6.1, throwaway venv) | sdist and wheel built; both PASSED. Nothing was uploaded |

## Paths executed on this machine

| Path | Ran here | Through |
|---|---|---|
| scalar table | yes | selftest, pytest, `simd_tests` |
| AVX2 table (FMA, F16C) | yes | selftest, pytest, `simd_tests` |
| AVX-VNNI: INT8 GEMM, Q8_0 and Q4_0 GEMV | yes | the path the avx2 table selects on this CPU |
| AVX2 `vpmaddubsw` Q8_0 / Q4_0 rows (no VNNI) | yes, C++ tests only | `simd_tests` calls these rows directly. On this CPU the dispatcher picks AVX-VNNI, so selftest and pytest do not reach them |
| INT8 `fp32_convert` (no VNNI) | yes | pytest (`_sgemm_u8s8_convert`) and `simd_tests` |
| AVX-512 table | **no** | the CPU has no AVX-512, and SDE is not installed |
| AVX512-VNNI | **no** | `simd_tests`: "avx512_vnni: SKIP (CPU has no AVX512_VNNI)" |
| BF16-dot (`vdpbf16ps`) | **no** | `simd_tests` runs its forced-on block only on CPUs with AVX512_BF16 |
| OpenMP threaded GEMM and GEMV | yes | OpenMP build, rows 11–12 (the default build skips these 74 tests) |

**Not executed anywhere locally:** the AVX-512 table, AVX512-VNNI and
BF16-dot (`vdpbf16ps`). On this CPU they need Intel SDE (`-skx`, `-spr`, see
[setup.md](setup.md)), which is not installed here.

## Same commit in CI

GitHub Actions [CI run 44696606255](https://github.com/Mattral/simd-ml-kernels/actions/runs/44696606255)
on `9b3a8088e77d` ran selftest and pytest on runners whose CPUs have AVX-512:
Intel Xeon Platinum 8573C, AMD EPYC 9V74 and Intel Xeon 6975P-C. On each,
`info` reported `runtime_isa` avx512 and `u8s8_path` / `q8_0_path`
avx512_vnni, so the AVX-512 table and the AVX512-VNNI path ran there and
passed. GitHub picks runner CPUs, so a later run may land on a CPU without
AVX-512.

The CI logs do not show that BF16-dot ran. Only the C++ test forces it on, and
ctest prints that test's output only when it fails. The C++ jobs also do not
record their CPU.

The `Linux / GCC-12 / OpenMP GEMM` job in `build-and-test.yml` builds the
package with `SIMD_ML_OPENMP=ON`, checks that `build_info()["openmp"]` is
true, and runs selftest and the full pytest suite, so the OpenMP-only tests
run in CI on every push and pull request. On `6459947f4570`
([Build and Test run 44854452272](https://github.com/Mattral/simd-ml-kernels/actions/runs/44854452272))
it reported 1584 passed, 41 skipped on an Intel Xeon Platinum 8573C (scalar,
avx2 and avx512 tables).

The Llama-3.2-1B parity tests (row 13) need the gated weights and are not run
in CI; this record is their only run for 0.4.0.
