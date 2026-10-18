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


def _throttled(*rs: dict | None) -> bool:
    """A case whose clock samples stayed below --min-ghz in every attempt
    (the core was throttled, or WSL2 moved the vCPU to an E-core): its
    numbers stay in the JSON but are not published in a table."""
    return any(r is not None and r["core_ghz"].get("below_min_ghz") for r in rs)


def _excluded_note(doc: dict, cases: list[tuple[str, dict]]) -> str:
    if not cases:
        return ""
    min_ghz = doc["protocol"].get("min_ghz")
    items = ", ".join(f"{name} (median {r['core_ghz']['median']:.2f}, min {r['core_ghz']['min']:.2f} GHz, "
                      f"{r['core_ghz']['attempts']} attempts)" for name, r in cases)
    return (f"\nNot shown: {items}. A clock sample stayed below the {min_ghz} GHz P-core threshold "
            f"(`--min-ghz`) in every attempt, so these numbers are not published; they remain in the JSON.")


def gemm_table(doc: dict, detail: bool) -> str:
    rows = ["| N | simd-ml-kernels GFLOP/s | OpenBLAS GFLOP/s | % of OpenBLAS |"
            + (" flop/cycle (ours / OpenBLAS) | IQR ms (ours) | IQR ms (OpenBLAS) |" if detail else "")
            + " core GHz median (min–max) |",
            "|---:|---:|---:|---:|" + ("---:|---:|---:|" if detail else "") + "---:|"]
    excluded = []
    for shape, o, b in sorted(_pairs(doc, "sgemm")):
        if _throttled(o, b):
            excluded.append((f"N = {shape[0]}", o))
            continue
        line = f"| {shape[0]} | {o['gflops']:.1f} | {b['gflops']:.1f} | {100 * o['gflops'] / b['gflops']:.0f}% |"
        if detail:
            line += (f" {o['flop_per_cycle']:.1f} / {b['flop_per_cycle']:.1f} |"
                     f" {o['p25_s'] * 1e3:.3f}–{o['p75_s'] * 1e3:.3f} |"
                     f" {b['p25_s'] * 1e3:.3f}–{b['p75_s'] * 1e3:.3f} |")
        rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded)


ACT_ORDER = ["gelu_exact", "gelu_tanh", "silu", "relu", "softmax", "layer_norm"]


def act_table(doc: dict, detail: bool) -> str:
    rows = ["| kernel | shape | simd-ml-kernels Gelem/s | elem/cycle | PyTorch Gelem/s | ours / PyTorch |"
            + (" IQR µs (ours) | IQR µs (PyTorch) |" if detail else "") + " core GHz median (min–max) |",
            "|---|---|---:|---:|---:|---:|" + ("---:|---:|" if detail else "") + "---:|"]
    excluded = []
    for k in ACT_ORDER:
        for shape, o, b in _pairs(doc, k):
            if o is None:
                continue
            shp = "×".join(str(s) for s in shape)
            if _throttled(o, b):
                excluded.append((f"`{k}` {shp}", o))
                continue
            tb = f"{b['gelem_per_s']:.3f}" if b else "not measured"
            ratio = f"{o['gelem_per_s'] / b['gelem_per_s']:.2f}×" if b else "—"
            line = f"| `{k}` | {shp} | {o['gelem_per_s']:.3f} | {o['elem_per_cycle']:.3f} | {tb} | {ratio} |"
            if detail:
                bi = f"{b['p25_s'] * 1e6:.1f}–{b['p75_s'] * 1e6:.1f}" if b else "—"
                line += f" {o['p25_s'] * 1e6:.1f}–{o['p75_s'] * 1e6:.1f} | {bi} |"
            rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded)


def section(docs: list[dict], detail: bool, link_prefix: str, latest_only: bool) -> str:
    """One block per result file, newest first; latest_only keeps the newest
    file per machine label (README), otherwise every run is listed (BENCHMARKS)."""
    if not docs:
        return "_No committed benchmark results._\n"
    docs = sorted(docs, key=lambda d: d["timestamp_utc"], reverse=True)
    if latest_only:
        seen: set[str] = set()
        docs = [d for d in docs
                if not (d["machine"]["label"] in seen or seen.add(d["machine"]["label"]))]
    out = []
    for d in docs:
        out.append(f"#### {d['machine']['label']}, {d['timestamp_utc'][:10]}, "
                   f"commit `{d['build']['git_commit']}`\n")
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
        "README.md": ("results-summary",
                      section(docs, detail=False, link_prefix="", latest_only=True)),
        "docs/BENCHMARKS.md": ("results-detail",
                               section(docs, detail=True, link_prefix="../", latest_only=False)),
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
