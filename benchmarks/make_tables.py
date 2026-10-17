"""Generate the benchmark tables in README.md and docs/BENCHMARKS.md from the
committed JSON in benchmarks/results/ (written by `python -m simd_ml_kernels
bench --json`). Tables are never edited by hand.

    python benchmarks/make_tables.py           # rewrite the generated sections
    python benchmarks/make_tables.py --check   # exit 1 if they are out of date (CI)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = "simd-ml-kernels-bench/1"
OURS = "simd_ml_kernels"


def load() -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "benchmarks", "results", "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("schema") == SCHEMA:
            d["_file"] = os.path.relpath(path, ROOT).replace(os.sep, "/")
            docs.append(d)
    return docs


def _pairs(doc: dict, kernel: str):
    """(shape, ours, baseline) for every case of `kernel` in doc."""
    by_shape: dict[tuple, dict] = {}
    for r in doc["results"]:
        if r["kernel"] == kernel:
            by_shape.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
    for shape, impls in by_shape.items():
        ours = impls.get(OURS)
        base = next((v for k, v in impls.items() if k != OURS), None)
        yield shape, ours, base


def _source(doc: dict, link_prefix: str) -> str:
    m, b = doc["machine"], doc["build"]
    blas = doc["baselines"]["blas"]
    torch = doc["baselines"]["torch"] or "not installed: not measured"
    return (f"{m['cpu_model']}, {m['core']}; Python {doc['python']}, "
            f"{b['compiler']}, commit `{b['git_commit']}`; baselines: NumPy {doc['baselines']['numpy']} "
            f"with {blas['name']} {blas['version']}, PyTorch {torch}; "
            f"{doc['protocol']['reps']} interleaved reps after {doc['protocol']['warmup']} warm-up; "
            f"{doc['timestamp_utc']}; [`{doc['_file']}`]({link_prefix}{doc['_file']})")


def _clk(r: dict) -> str:
    c = r["core_ghz"]
    return f"{c['median']:.2f} ({c['min']:.2f}–{c['max']:.2f})"


def gemm_table(doc: dict, detail: bool) -> str:
    rows = ["| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS |"
            + (" flop/cycle (ours / OpenBLAS) | IQR ms (ours) | IQR ms (OpenBLAS) | core GHz median (min–max) |" if detail else ""),
            "|---:|---:|---:|---:|" + ("---:|---:|---:|---:|" if detail else "")]
    for shape, o, b in sorted(_pairs(doc, "sgemm")):
        line = f"| {shape[0]} | {o['gflops']:.1f} | {b['gflops']:.1f} | {100 * o['gflops'] / b['gflops']:.0f}% |"
        if detail:
            line += (f" {o['flop_per_cycle']:.1f} / {b['flop_per_cycle']:.1f} |"
                     f" {o['p25_s'] * 1e3:.3f}–{o['p75_s'] * 1e3:.3f} |"
                     f" {b['p25_s'] * 1e3:.3f}–{b['p75_s'] * 1e3:.3f} | {_clk(o)} |")
        rows.append(line)
    return "\n".join(rows)


ACT_ORDER = ["gelu_exact", "gelu_tanh", "silu", "relu", "softmax", "layer_norm"]


def act_table(doc: dict, detail: bool) -> str:
    rows = ["| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch |"
            + (" IQR µs (ours) | IQR µs (PyTorch) | core GHz median (min–max) |" if detail else ""),
            "|---|---|---:|---:|---:|---:|" + ("---:|---:|---:|" if detail else "")]
    for k in ACT_ORDER:
        for shape, o, b in _pairs(doc, k):
            if o is None:
                continue
            shp = "×".join(str(s) for s in shape)
            tb = f"{b['gelem_per_s']:.3f}" if b else "not measured"
            ratio = f"{o['gelem_per_s'] / b['gelem_per_s']:.2f}×" if b else "—"
            line = f"| `{k}` | {shp} | {o['gelem_per_s']:.3f} | {o['elem_per_cycle']:.3f} | {tb} | {ratio} |"
            if detail:
                bi = f"{b['p25_s'] * 1e6:.1f}–{b['p75_s'] * 1e6:.1f}" if b else "—"
                line += f" {o['p25_s'] * 1e6:.1f}–{o['p75_s'] * 1e6:.1f} | {bi} | {_clk(o)} |"
            rows.append(line)
    return "\n".join(rows)


def section(docs: list[dict], detail: bool, link_prefix: str) -> str:
    if not docs:
        return "_No committed benchmark results._\n"
    out = []
    for d in docs:
        out.append(f"#### {d['machine']['label']}\n")
        out.append("Source: " + _source(d, link_prefix) + ".\n")
        out.append("**FP32 GEMM, square, single thread**\n")
        out.append(gemm_table(d, detail) + "\n")
        out.append("**Activations and normalization, out-of-place, single thread**\n")
        out.append(act_table(d, detail) + "\n")
    return "\n".join(out).rstrip() + "\n"


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker '{name}' not found")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    docs = load()
    targets = {
        "README.md": ("results-summary", section(docs, detail=False, link_prefix="")),
        "docs/BENCHMARKS.md": ("results-detail", section(docs, detail=True, link_prefix="../")),
    }
    stale = []
    for rel, (name, body) in targets.items():
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as f:
            old = f.read()
        new = replace_block(old, name, body)
        if new != old:
            stale.append(rel)
            if not args.check:
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(new)
    if args.check and stale:
        print("out of date (run python benchmarks/make_tables.py):", ", ".join(stale))
        return 1
    print("updated: " + (", ".join(stale) if stale else "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
