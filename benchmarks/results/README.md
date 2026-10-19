# Benchmark results

One JSON file per benchmark run, written by

```bash
python -m simd_ml_kernels bench --json benchmarks/results/<label>_<date>_<commit>.json \
    --label <label> --core "<core>" --note "<power settings>"
```

following the protocol in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md). Every
benchmark number in README.md and docs/BENCHMARKS.md is generated from these
files by `benchmarks/make_tables.py`; CI checks the tables are up to date.
Do not edit result files by hand.

## Schema `simd-ml-kernels-bench/1`

| key | contents |
|---|---|
| `schema` | `"simd-ml-kernels-bench/1"` |
| `machine` | `label`, `cpu_model`, `core` (caller's description of the pinned core), `pinned_cpus`, `caches` (CPUID leaf 4 of the pinned core: level, type, size_kib, ways, line_bytes, max_sharing_cpus), `logical_cpus`, `os`, `note`, `stream_{read,triad}_gbs` and `stream_{read,triad}_allcore_gbs` (bandwidth roofs; null where not measured) |
| `build` | `version`, `git_commit`, `compiler`, `kernel_flags`, `isa_compiled`, `openmp`, `sgemm_tile`, `sgemm_blocking` {`kc`, `mc`, `nc`}, `u8s8_path` (from `build_info()`) |
| `baselines` | `numpy`, `blas` {`name`, `version`}, `torch` (null if not installed) |
| `env` | `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `SIMD_ML_ISA`, `SIMD_ML_BF16_DOT` |
| `protocol` | `warmup`, `reps`, `statistic`, `interleaved`, `clock`, `min_ghz`, `malloc`, `stream`, `gemv` |
| `command`, `python`, `timestamp_utc` | how and when the file was produced |
| `results[]` | `kernel`, `impl`, `shape`, `median_s`, `p25_s`, `p75_s`, `min_s`, `reps`, `core_ghz` {`median`, `min`, `max`, `samples`, `attempts`, `below_min_ghz`}, and `gflops` + `flop_per_cycle` (GEMM; the GEMV cases also `bytes`, `gb_per_s`, `pct_stream_read`, `pct_stream_triad`, `weight_copies`), `gelem_per_s` + `elem_per_cycle` (element-wise) or `gb_per_s` (`stream_*` rows) |

Older files lack the keys added later (roofs, GEMV fields, `malloc`, the
`SIMD_ML_*` variables). The full key list is also in
[AGENTS.md](../../AGENTS.md#results-and-their-schema).
