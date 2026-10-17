#!/usr/bin/env bash
# benchmarks/run_bench.sh — Build and run the simd-ml-kernels benchmark suite.
#
# Runs both:
#   bench       — cycle-accurate RDTSC harness (quick check)
#   bench_stat  — statistical wall-clock harness (30 reps, 95% CI, JSON output)
#
# For reproducible results, lock CPU frequency before running:
#   sudo cpupower frequency-set -g performance
#   taskset -c 0 ./benchmarks/run_bench.sh
#
# Usage:
#   ./benchmarks/run_bench.sh [--reps N] [--sizes N,M,...] [--no-stat]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"

# ─── Parse arguments ─────────────────────────────────────────────────────────
STAT_REPS=30
STAT_SIZES="64,128,256,512"
RUN_STAT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps)    STAT_REPS="$2";  shift 2 ;;
    --sizes)   STAT_SIZES="$2"; shift 2 ;;
    --no-stat) RUN_STAT=0;      shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ─── Build ────────────────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR" "$RESULTS_DIR"
cd "$BUILD_DIR"

cmake "$REPO_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja \
    -DSIMD_ML_OPENMP=OFF 2>&1 | grep -E "^-- |error:" || true

cmake --build . --target bench bench_stat --parallel

printf "\n╔══════════════════════════════════════════════╗\n"
printf "║  simd-ml-kernels Benchmark                  ║\n"
printf "╚══════════════════════════════════════════════╝\n"

# ─── Cycle-accurate quick run ─────────────────────────────────────────────────
printf "\n=== Cycle-accurate RDTSC harness (bench) ===\n"
"$BUILD_DIR/bench" --warmup 5 --reps 10 \
    --json "$RESULTS_DIR/bench_rdtsc.json"

# ─── Statistical harness ─────────────────────────────────────────────────────
if [[ "$RUN_STAT" -eq 1 ]]; then
  printf "\n=== Statistical harness (bench_stat, %d reps) ===\n" "$STAT_REPS"
  taskset_cmd=""
  if command -v taskset &>/dev/null; then
    taskset_cmd="taskset -c 0"
    printf "  (pinned to core 0 via taskset)\n"
  fi

  $taskset_cmd "$BUILD_DIR/bench_stat" \
      --reps "$STAT_REPS" \
      --sizes "$STAT_SIZES" \
      --output "$RESULTS_DIR/bench_stat.json"

  printf "\nResults written to %s\n" "$RESULTS_DIR/bench_stat.json"
fi

printf "\nBenchmark complete.\n"
