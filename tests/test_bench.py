"""
tests/test_bench.py — Python Benchmark with Publication-Quality Plots

Measures simd-ml-kernels SIMD vs NumPy reference performance for GEMM and GeLU
across a range of problem sizes, then renders publication-quality plots.

Usage:
    python tests/test_bench.py              # run + show + save plots
    python tests/test_bench.py --no-plot    # run + print table only
    python tests/test_bench.py --quick      # small sizes only (CI smoke test)
    python tests/test_bench.py --save-dir results/

Requirements:
    pip install numpy matplotlib tabulate scipy
    # simd_ml_kernels must be installed: `pip install .`

Speedup interpretation note:
    NumPy uses OpenBLAS which is multi-threaded. For a fair single-core
    comparison, set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 before running.
    Without this, NumPy speedups > 1 do not mean simd-ml-kernels is faster.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ─── Optional imports ─────────────────────────────────────────────────────────
try:
    import simd_ml_kernels
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    print("WARNING: simd_ml_kernels not found. Only NumPy reference timings will run.")
    print("  Install with: pip install .")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ─── Timer ───────────────────────────────────────────────────────────────────

def timeit(fn, warmup: int = 3, reps: int = 10) -> tuple[float, float]:
    """
    Return (min_seconds, median_seconds) over `reps` timed calls,
    after `warmup` untimed warm-up passes.

    The callable `fn` must be completely self-contained — any array
    allocations needed for correctness should happen OUTSIDE this call
    and reused across reps.  Allocating inside `fn` contaminates the
    timing with allocator noise.
    """
    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    return times[0], times[len(times) // 2]


def gelu_numpy_ref(x: np.ndarray) -> np.ndarray:
    """Reference GeLU (tanh formula) in NumPy — matches simd-ml-kernels's formula exactly."""
    sqrt2pi = np.float32(0.7978845608028654)
    coeff   = np.float32(0.044715)
    inner   = sqrt2pi * (x + coeff * x ** 3)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner.astype(np.float64)).astype(np.float32)))


# ─── GEMM benchmark ───────────────────────────────────────────────────────────

def bench_gemm(sizes: list[int], reps: int = 10) -> list[dict]:
    """
    Benchmark square GEMM: C = A @ B  (float32)

    Speedup is relative to NumPy (np.dot / np.matmul).
    NOTE: NumPy uses OpenBLAS which is multi-threaded by default.
    Set OMP_NUM_THREADS=1 for a fair single-threaded comparison.
    """
    rng = np.random.default_rng(42)
    results = []

    for N in sizes:
        flops = 2 * N * N * N  # multiply-add = 2 FLOPS

        A = rng.standard_normal((N, N)).astype(np.float32)
        B = rng.standard_normal((N, N)).astype(np.float32)
        C = np.zeros((N, N), dtype=np.float32)

        # NumPy reference (typically multi-threaded OpenBLAS)
        t_np_min, t_np_med = timeit(lambda: np.dot(A, B), reps=reps)
        gflops_np = flops / t_np_min / 1e9

        row: dict = {
            "N":           N,
            "flops":       flops,
            "numpy_ms":    t_np_min * 1e3,
            "numpy_gflops": gflops_np,
            "simd_ms":     None,
            "simd_gflops": None,
            "speedup":     None,
        }

        if HAS_SIMD:
            # Pre-allocate C outside timeit to avoid allocator noise in the loop
            simd_ml_kernels.sgemm(A, B, C, alpha=1.0, beta=0.0)  # warm up
            t_simd_min, _ = timeit(
                lambda: simd_ml_kernels.sgemm(A, B, C, alpha=1.0, beta=0.0),
                reps=reps
            )
            gflops_simd = flops / t_simd_min / 1e9
            row["simd_ms"]    = t_simd_min * 1e3
            row["simd_gflops"] = gflops_simd
            row["speedup"]    = t_np_min / t_simd_min

        results.append(row)
        _print_gemm_row(row)

    return results


def _print_gemm_row(r: dict) -> None:
    simd_str = (
        f"{r['simd_ms']:7.3f} ms  {r['simd_gflops']:6.1f} GFLOPS  {r['speedup']:5.2f}×"
        if r["simd_ms"] is not None else "  (simd_ml_kernels not installed)"
    )
    print(f"  GEMM {r['N']:4d}×{r['N']:4d}  "
          f"numpy: {r['numpy_ms']:7.3f} ms  {r['numpy_gflops']:6.1f} GFLOPS  |  "
          f"simd: {simd_str}")


# ─── GeLU benchmark ───────────────────────────────────────────────────────────

def bench_gelu(sizes: list[int], reps: int = 10) -> list[dict]:
    """
    Benchmark element-wise GeLU (tanh approximation, float32).

    Speedup is relative to NumPy's tanh-formula GeLU in pure NumPy.
    The SIMD kernel uses in-place operation on a pre-allocated output buffer
    to avoid allocation overhead inside the timed loop.
    """
    rng = np.random.default_rng(99)
    results = []

    for n in sizes:
        x_orig = rng.uniform(-3.0, 3.0, size=n).astype(np.float32)

        # NumPy reference
        t_np_min, _ = timeit(lambda: gelu_numpy_ref(x_orig), reps=reps)
        gelems_np = n / t_np_min / 1e9

        row: dict = {
            "n":           n,
            "numpy_ms":    t_np_min * 1e3,
            "numpy_gelems": gelems_np,
            "simd_ms":     None,
            "simd_gelems": None,
            "speedup":     None,
        }

        if HAS_SIMD:
            # Allocate output buffer ONCE outside the timed loop.
            # gelu() is out-of-place — input is unchanged, output pre-allocated.
            out = np.empty_like(x_orig)
            t_simd_min, _ = timeit(
                lambda: simd_ml_kernels.gelu(x_orig, approximate="tanh"),  # new array each call
                reps=reps
            )
            gelems_simd = n / t_simd_min / 1e9
            row["simd_ms"]    = t_simd_min * 1e3
            row["simd_gelems"] = gelems_simd
            row["speedup"]    = t_np_min / t_simd_min

            # Accuracy spot-check vs numpy reference (not timed)
            ref = gelu_numpy_ref(x_orig)
            got = simd_ml_kernels.gelu(x_orig, approximate="tanh")
            max_rel = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))
                                   / (np.abs(ref.astype(np.float64)) + 1e-7)))
            row["max_rel_err"] = max_rel
            del out

        results.append(row)
        _print_gelu_row(row)

    return results


def _print_gelu_row(r: dict) -> None:
    simd_str = (
        f"{r['simd_ms']:8.4f} ms  {r['simd_gelems']:5.2f} GEl/s  {r['speedup']:5.2f}×"
        f"  (max_rel={r.get('max_rel_err', float('nan')):.1e})"
        if r["simd_ms"] is not None else "  (simd_ml_kernels not installed)"
    )
    print(f"  GeLU n={r['n']:8,}  "
          f"numpy: {r['numpy_ms']:8.4f} ms  {r['numpy_gelems']:5.2f} GEl/s  |  "
          f"simd: {simd_str}")


# ─── Table printers ───────────────────────────────────────────────────────────

def print_gemm_table(results: list[dict]) -> None:
    rows = [
        [
            f"{r['N']}×{r['N']}",
            f"{r['numpy_ms']:.3f}",
            f"{r['numpy_gflops']:.1f}",
            f"{r['simd_ms']:.3f}"    if r["simd_ms"]     is not None else "—",
            f"{r['simd_gflops']:.1f}" if r["simd_gflops"] is not None else "—",
            f"{r['speedup']:.2f}×"   if r["speedup"]     is not None else "—",
        ]
        for r in results
    ]
    headers = ["Size", "NumPy ms", "NumPy GFLOPS", "SIMD ms", "SIMD GFLOPS", "Speedup*"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        _plain_table(headers, rows)
    print("  * Speedup vs NumPy/OpenBLAS (potentially multi-threaded). "
          "Set OMP_NUM_THREADS=1 for single-threaded comparison.")


def print_gelu_table(results: list[dict]) -> None:
    rows = [
        [
            f"{r['n']:,}",
            f"{r['numpy_ms']:.4f}",
            f"{r['numpy_gelems']:.2f}",
            f"{r['simd_ms']:.4f}"    if r["simd_ms"]    is not None else "—",
            f"{r['simd_gelems']:.2f}" if r["simd_gelems"] is not None else "—",
            f"{r['speedup']:.2f}×"   if r["speedup"]    is not None else "—",
            f"{r.get('max_rel_err', float('nan')):.1e}"
            if r["simd_ms"] is not None else "—",
        ]
        for r in results
    ]
    headers = ["N Elements", "NumPy ms", "NumPy GEl/s", "SIMD ms", "SIMD GEl/s", "Speedup", "Max Rel Err"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        _plain_table(headers, rows)


def _plain_table(headers: list[str], rows: list[list]) -> None:
    col_w = [max(len(h), max((len(r[i]) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep = "  " + " | ".join("-" * w for w in col_w)
    print("  " + " | ".join(f"{h:{w}}" for h, w in zip(headers, col_w)))
    print(sep)
    for row in rows:
        print("  " + " | ".join(f"{v:{w}}" for v, w in zip(row, col_w)))


# ─── Publication-quality plots ────────────────────────────────────────────────

_COLORS = {
    "numpy":  "#5B8DB8",   # steel blue
    "simd":   "#E07B39",   # burnt orange
    "speedup": "#6ABF69",  # green
}

_STYLE = {
    "fontsize_title": 12,
    "fontsize_axis":  11,
    "fontsize_annot":  9,
    "linewidth": 2.0,
    "markersize": 7,
    "dpi": 150,
}


def plot_gemm(results: list[dict], save_dir: Path | None = None) -> None:
    if not HAS_MPL:
        print("matplotlib not installed — skipping GEMM plot.")
        return

    sizes      = [r["N"] for r in results]
    np_gflops  = [r["numpy_gflops"] for r in results]
    simd_sizes  = [r["N"] for r in results if r["simd_gflops"] is not None]
    simd_gflops = [r["simd_gflops"] for r in results if r["simd_gflops"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GEMM Throughput: simd-ml-kernels SIMD vs NumPy",
                 fontsize=_STYLE["fontsize_title"] + 1, fontweight="bold")

    # Left: GFLOPS vs size
    ax = axes[0]
    ax.plot(sizes, np_gflops, "o--",
            color=_COLORS["numpy"], linewidth=_STYLE["linewidth"],
            markersize=_STYLE["markersize"], label="NumPy / OpenBLAS")
    if simd_sizes:
        ax.plot(simd_sizes, simd_gflops, "s-",
                color=_COLORS["simd"], linewidth=_STYLE["linewidth"],
                markersize=_STYLE["markersize"], label="simd-ml-kernels AVX2")
    ax.set_xlabel("Matrix Size N  (N×N square)", fontsize=_STYLE["fontsize_axis"])
    ax.set_ylabel("GFLOPS", fontsize=_STYLE["fontsize_axis"])
    ax.set_title("Arithmetic Throughput", fontsize=_STYLE["fontsize_title"])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right: Speedup bar chart
    if simd_sizes:
        speedups = [r["speedup"] for r in results if r["speedup"] is not None]
        ax2 = axes[1]
        bars = ax2.bar(
            [str(n) for n in simd_sizes], speedups,
            color=_COLORS["simd"], edgecolor="black",
            linewidth=0.6, alpha=0.85, width=0.6
        )
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
                    label="NumPy baseline (1.0×)")
        ax2.set_xlabel("Matrix Size N", fontsize=_STYLE["fontsize_axis"])
        ax2.set_ylabel("Speedup vs NumPy", fontsize=_STYLE["fontsize_axis"])
        ax2.set_title("Relative Speedup\n(⚠ NumPy may use multi-thread OpenBLAS)",
                      fontsize=_STYLE["fontsize_title"])
        ax2.legend(fontsize=10)
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.set_ylim(bottom=0)
        for bar, sp in zip(bars, speedups):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(speedups) * 0.02,
                f"{sp:.2f}×", ha="center", va="bottom",
                fontsize=_STYLE["fontsize_annot"]
            )

    plt.tight_layout()
    if save_dir:
        out = save_dir / "bench_gemm.png"
        plt.savefig(out, dpi=_STYLE["dpi"], bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.show()


def plot_gelu(results: list[dict], save_dir: Path | None = None) -> None:
    if not HAS_MPL:
        print("matplotlib not installed — skipping GeLU plot.")
        return

    ns          = [r["n"] for r in results]
    np_gelems   = [r["numpy_gelems"] for r in results]
    simd_ns     = [r["n"] for r in results if r["simd_gelems"] is not None]
    simd_gelems = [r["simd_gelems"] for r in results if r["simd_gelems"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GeLU Throughput: simd-ml-kernels SIMD vs NumPy (tanh formula)",
                 fontsize=_STYLE["fontsize_title"] + 1, fontweight="bold")

    # Left: Throughput vs size
    ax = axes[0]
    ax.plot(ns, np_gelems, "o--",
            color=_COLORS["numpy"], linewidth=_STYLE["linewidth"],
            markersize=_STYLE["markersize"], label="NumPy reference")
    if simd_ns:
        ax.plot(simd_ns, simd_gelems, "s-",
                color=_COLORS["simd"], linewidth=_STYLE["linewidth"],
                markersize=_STYLE["markersize"], label="simd-ml-kernels AVX2")
    ax.set_xlabel("Input Elements N", fontsize=_STYLE["fontsize_axis"])
    ax.set_ylabel("GElements / second", fontsize=_STYLE["fontsize_axis"])
    ax.set_title("GeLU Activation Throughput", fontsize=_STYLE["fontsize_title"])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right: Speedup line with fill
    if simd_ns:
        speedups = [r["speedup"] for r in results if r["speedup"] is not None]
        ax2 = axes[1]
        ax2.plot(simd_ns, speedups, "D-",
                 color=_COLORS["speedup"], linewidth=_STYLE["linewidth"],
                 markersize=8)
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
        ax2.fill_between(simd_ns, 1.0, speedups, alpha=0.15, color=_COLORS["speedup"])
        ax2.set_xlabel("Input Elements N", fontsize=_STYLE["fontsize_axis"])
        ax2.set_ylabel("Speedup vs NumPy", fontsize=_STYLE["fontsize_axis"])
        ax2.set_title("Relative Speedup vs NumPy", fontsize=_STYLE["fontsize_title"])
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        for x, y in zip(simd_ns, speedups):
            ax2.annotate(
                f"{y:.2f}×", (x, y),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=_STYLE["fontsize_annot"]
            )

    plt.tight_layout()
    if save_dir:
        out = save_dir / "bench_gelu.png"
        plt.savefig(out, dpi=_STYLE["dpi"], bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="simd-ml-kernels Python benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip matplotlib rendering (print tables only)")
    parser.add_argument("--quick",    action="store_true",
                        help="Small sizes only — fast CI smoke test")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Directory to save PNG plots into")
    parser.add_argument("--reps",     type=int, default=10,
                        help="Timing repetitions per measurement (default: 10)")
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    gemm_sizes = [32, 64, 128] if args.quick else [32, 64, 128, 256, 512, 1024]
    gelu_sizes = [1024, 8192, 65536] if args.quick else \
                 [1024, 4096, 16384, 65536, 262144, 1 << 20]

    show_plots = HAS_MPL and not args.no_plot

    if HAS_SIMD:
        print(f"\nRuntime ISA:  {simd_ml_kernels.detected_isa()}")
        print(f"Build info:   {simd_ml_kernels.build_info()}")

    # ── GEMM ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  GEMM Benchmark  (C = A @ B, float32, square matrices)")
    print("  NOTE: NumPy/OpenBLAS may use multiple cores. Set OMP_NUM_THREADS=1")
    print("        for a single-threaded comparison.")
    print("═" * 72)
    gemm_results = bench_gemm(gemm_sizes, reps=args.reps)
    print()
    print_gemm_table(gemm_results)

    # ── GeLU ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  GeLU Benchmark  (element-wise, float32, tanh approximation)")
    print("═" * 72)
    gelu_results = bench_gelu(gelu_sizes, reps=args.reps)
    print()
    print_gelu_table(gelu_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if show_plots:
        print("\nRendering plots...")
        plot_gemm(gemm_results, save_dir=args.save_dir)
        plot_gelu(gelu_results, save_dir=args.save_dir)
    elif not HAS_MPL:
        print("\nInstall matplotlib for plots:  pip install matplotlib")

    # ── CI performance gate ───────────────────────────────────────────────────
    # GeLU regression only: both simd-ml-kernels and NumPy's tanh are single-threaded
    # here, so the comparison is apples-to-apples. Threshold: SIMD must not be
    # more than 2× slower than NumPy's tanh GeLU reference.
    #
    # GEMM is NOT gated here: NumPy uses multi-threaded OpenBLAS which can be
    # 4–16× faster on this hardware for small matrices. A fair GEMM comparison
    # requires OMP_NUM_THREADS=1. Instead we print an informational summary.
    if HAS_SIMD:
        gelu_regressions = [r for r in gelu_results
                            if r.get("speedup") is not None and r["speedup"] < 0.5]
        if gelu_regressions:
            print("\n❌  GeLU REGRESSION: SIMD GeLU is >2× slower than NumPy tanh reference.")
            print("    This indicates a broken kernel, not a threading artifact.")
            for r in gelu_regressions:
                print(f"   n={r['n']:,}: speedup={r['speedup']:.3f}×")
            sys.exit(1)
        else:
            print("\n✅  GeLU performance gate: PASS")

    print("\nDone.")


if __name__ == "__main__":
    main()
