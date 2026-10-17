#!/usr/bin/env python3
"""
check_regression.py — CI Regression Gating for simd-ml-kernels Benchmarks

Compares a current benchmark JSON result against a committed baseline.
Fails (exit code 1) if any kernel's throughput dropped by more than
--max-regression-pct percent versus the baseline.

Usage:
    python benchmarks/check_regression.py \\
        --baseline benchmarks/results/gemm_results.json \\
        --current  benchmarks/results/ci_gemm_results.json \\
        --max-regression-pct 20

JSON schema (matches bench_stat.cpp output, field names as of v0.2.0):
{
  "gemm": [
    {
      "kernel": "simd_packed",
      "N": 256,
      "min_ms": 0.572,
      "mean_ms": 0.590,
      "median_ms": 0.587,
      "stddev_ms": 0.015,
      "ci95_half_ms": 0.006,
      "gflops_min_latency": 58.6,
      "gflops_mean_latency": 56.8
    }
  ],
  "gelu": [
    {
      "kernel": "gelu_avx2",
      "n": 1048576,
      "min_ms": 0.876,
      "gelems_per_sec": 1.196
    }
  ]
}

Field name history:
  v0.1.x: gflops_peak, gflops_mean          (deprecated)
  v0.2.0: gflops_min_latency, gflops_mean_latency  (current)

check_regression.py accepts both old and new field names for backward
compatibility when comparing a v0.1 baseline against a v0.2 current run.
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def get_gemm_perf(entry: dict) -> float:
    """Extract GEMM throughput, accepting both old and new field names.

    Uses gflops_min_latency (best-case, cache-warm) rather than
    gflops_mean_latency because min-latency is more reproducible on noisy
    CI runners — the mean fluctuates with OS scheduler jitter, but the
    minimum captures genuine hardware peak before interference.
    """
    # Prefer min_latency (most reproducible); fall back for backward compat
    for key in ("gflops_min_latency", "gflops_peak",
                "gflops_mean_latency", "gflops_mean"):
        if key in entry:
            return float(entry[key])
    return 0.0


def get_gelu_perf(entry: dict) -> float:
    return float(entry.get("gelems_per_sec", 0.0))


def build_key(entry: dict, kind: str) -> str:
    """Build a unique key for a benchmark entry."""
    if kind == "gemm":
        return f"{entry['kernel']}@N={entry['N']}"
    elif kind == "gelu":
        return f"{entry['kernel']}@n={entry['n']}"
    return str(entry)


def check_regressions(baseline: dict, current: dict,
                      max_regression_pct: float) -> int:
    """
    Returns the number of regressions detected.
    Prints a summary table to stdout.
    """
    regressions = 0
    total_checks = 0

    print(f"\n  {'Key':<42} {'Baseline':>10} {'Current':>10} {'Δ%':>8}  Status")
    print(f"  {'-'*42} {'-'*10} {'-'*10} {'-'*8}  ------")

    for kind in ("gemm", "gelu"):
        if kind not in baseline or kind not in current:
            continue

        get_perf = get_gemm_perf if kind == "gemm" else get_gelu_perf
        unit     = "GFLOPS"      if kind == "gemm" else "GEl/s"

        base_map: dict[str, float] = {}
        for entry in baseline[kind]:
            k = build_key(entry, kind)
            v = get_perf(entry)
            if v > 0:
                base_map[k] = v

        for entry in current[kind]:
            k       = build_key(entry, kind)
            cur_val = get_perf(entry)

            if k not in base_map:
                print(f"  {k:<42} {'N/A':>10} {cur_val:>10.2f} {'N/A':>8}  NEW ({unit})")
                continue

            base_val   = base_map[k]
            pct_change = 100.0 * (cur_val - base_val) / base_val
            total_checks += 1

            status = "✅ OK"
            if pct_change < -max_regression_pct:
                status = "❌ REGRESSION"
                regressions += 1
            elif pct_change > 15.0:
                status = "⬆ IMPROVED"

            print(f"  {k:<42} {base_val:>10.2f} {cur_val:>10.2f} {pct_change:>+8.1f}%  {status}")

    print()
    print(f"  Checked {total_checks} entries | "
          f"Regressions: {regressions} | "
          f"Threshold: -{max_regression_pct:.0f}%")

    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI regression checker for simd-ml-kernels benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--baseline", required=True,
                        help="Path to committed baseline JSON file")
    parser.add_argument("--current",  required=True,
                        help="Path to current CI run JSON file")
    parser.add_argument("--max-regression-pct", type=float, default=20.0,
                        help="Maximum allowed throughput drop in %% (default: 20)")
    args = parser.parse_args()

    print("simd-ml-kernels Benchmark Regression Check")
    print(f"  Baseline : {args.baseline}")
    print(f"  Current  : {args.current}")
    print(f"  Threshold: -{args.max_regression_pct:.0f}%% maximum drop")

    baseline = load_json(args.baseline)
    current  = load_json(args.current)

    regressions = check_regressions(baseline, current, args.max_regression_pct)

    if regressions > 0:
        print(f"\n❌  FAIL: {regressions} performance regression(s) detected.")
        print("   Investigate before merging. Consider:")
        print("   1. Noise? Re-run bench_stat --reps 50 on a quiet, pinned core.")
        print("   2. Real regression? Use perf stat or Intel VTune to profile.")
        sys.exit(1)
    else:
        print(f"\n✅  PASS: No regressions detected "
              f"(threshold: -{args.max_regression_pct:.0f}%)")
        sys.exit(0)


if __name__ == "__main__":
    main()
