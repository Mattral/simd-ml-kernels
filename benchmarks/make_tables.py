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


def _unstable(*rs: dict | None) -> bool:
    """A GEMV case where an implementation read faster than 105% of the case's
    own read roof (measured in the case): the roof was not reliable there."""
    return any(r is not None and r.get("roof_unstable") for r in rs)


def _unstable_note(cases: list[tuple[str, dict]]) -> str:
    if not cases:
        return ""
    items = ", ".join(f"{name} (roof {r['roof_read_gbs']:.1f} GB/s)" for name, r in cases)
    return (f"\nNot shown: {items}. An implementation ran above 105% of the read roof measured in the same "
            "case (`roof_unstable`), so the roof was not reliable there; the numbers remain in the JSON.")


def _excluded_note(doc: dict, cases: list[tuple[str, dict]]) -> str:
    if not cases:
        return ""
    min_ghz = doc["protocol"].get("min_ghz")
    items = ", ".join(f"{name} (median {r['core_ghz']['median']:.2f}, min {r['core_ghz']['min']:.2f} GHz, "
                      f"{r['core_ghz']['attempts']} attempts)" for name, r in cases)
    where = ("threshold on the slowest thread" if _is_threads(doc) else "P-core threshold")
    return (f"\nNot shown: {items}. A clock sample stayed below the {min_ghz} GHz {where} "
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


LOWP = [("sgemm_f16", "FP16"), ("sgemm_bf16", "BF16")]


def lowp_table(doc: dict, detail: bool) -> str:
    """FP16/BF16-input GEMM: ours, our FP32 GEMM on the same values, and
    PyTorch's matmul on float16/bfloat16 tensors (interleaved in one case)."""
    rows = ["| input | M×N×K | simd-ml-kernels GFLOP/s | our FP32 GFLOP/s | % of FP32 | PyTorch GFLOP/s | ours / PyTorch |"
            + (" IQR ms (ours) | IQR ms (FP32) | IQR ms (PyTorch) |" if detail else "")
            + " core GHz median (min–max) |",
            "|---|---|---:|---:|---:|---:|---:|" + ("---:|---:|---:|" if detail else "") + "---:|"]
    excluded = []
    for kernel, label in LOWP:
        by_shape: dict[tuple, dict] = {}
        for r in doc["results"]:
            if r["kernel"] == kernel:
                by_shape.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
        for shape in sorted(by_shape, key=lambda s: (s[0] != s[1], s)):
            impls = by_shape[shape]
            o, f = impls.get(OURS), impls.get(OURS + " fp32")
            t = next((v for k, v in impls.items() if k.startswith("torch")), None)
            if o is None or f is None:
                continue
            shp = "×".join(str(s) for s in shape)
            if _throttled(o, f, t):
                excluded.append((f"{label} {shp}", o))
                continue
            tg = f"{t['gflops']:.1f}" if t else "not measured"
            ratio = f"{o['gflops'] / t['gflops']:.2f}×" if t else "—"
            line = (f"| {label} | {shp} | {o['gflops']:.1f} | {f['gflops']:.1f} |"
                    f" {100 * o['gflops'] / f['gflops']:.0f}% | {tg} | {ratio} |")
            if detail:
                ti = f"{t['p25_s'] * 1e3:.3f}–{t['p75_s'] * 1e3:.3f}" if t else "—"
                line += (f" {o['p25_s'] * 1e3:.3f}–{o['p75_s'] * 1e3:.3f} |"
                         f" {f['p25_s'] * 1e3:.3f}–{f['p75_s'] * 1e3:.3f} | {ti} |")
            rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded)


def int8_table(doc: dict, detail: bool) -> str:
    """INT8 GEMM (sgemm_u8s8): ours on the active path, ours on the FP32
    conversion path, our FP32 GEMM and PyTorch's quantized linear, in one
    interleaved case per shape. Rates in GOP/s (2·M·N·K multiply-adds)."""
    rows = ["| M×N×K | simd-ml-kernels GOP/s | × our FP32 | FP32-conversion path GOP/s | × our FP32 | PyTorch GOP/s | ours / PyTorch |"
            + (" IQR ms (ours) | IQR ms (PyTorch) |" if detail else "") + " core GHz median (min–max) |",
            "|---|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if detail else "") + "---:|"]
    by_shape: dict[tuple, dict] = {}
    for r in doc["results"]:
        if r["kernel"] == "sgemm_u8s8":
            by_shape.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
    excluded = []
    for shape in sorted(by_shape, key=lambda s: (s[0] != s[1], s)):
        impls = by_shape[shape]
        o, f, cv = impls.get(OURS), impls.get(OURS + " fp32"), impls.get(OURS + " fp32_convert")
        t = next((v for k, v in impls.items() if k.startswith("torch")), None)
        if o is None or f is None or cv is None:
            continue
        shp = "×".join(str(s) for s in shape)
        if _throttled(o, f, cv, t):
            excluded.append((shp, o))
            continue
        tg = f"{t['gflops']:.1f}" if t else "not measured"
        ratio = f"{o['gflops'] / t['gflops']:.2f}×" if t else "—"
        line = (f"| {shp} | {o['gflops']:.1f} | {o['gflops'] / f['gflops']:.2f}× | {cv['gflops']:.1f} |"
                f" {cv['gflops'] / f['gflops']:.2f}× | {tg} | {ratio} |")
        if detail:
            ti = f"{t['p25_s'] * 1e3:.3f}–{t['p75_s'] * 1e3:.3f}" if t else "—"
            line += f" {o['p25_s'] * 1e3:.3f}–{o['p75_s'] * 1e3:.3f} | {ti} |"
        rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded)


GEMV = [("gemv_fp32", "FP32"), ("gemv_fp16", "FP16"), ("gemv_bf16", "BF16")]


def _pct(v) -> str:
    return f"{v:.0f}%" if v is not None else "not measured"


def gemv_table(doc: dict, detail: bool) -> str:
    """Decode GEMV y = x·Wᵀ: GB/s, % of the read-only stream (the roof for a
    read-only kernel) and of the STREAM triad, and speed against PyTorch's
    F.linear (and NumPy's sgemv for FP32) on the same weights."""
    per_case = doc["protocol"].get("gemv_roof") == "per case"
    rows = ["| weights | M×N×K | simd-ml-kernels µs | GB/s | % of read " + ("roof" if per_case else "stream")
            + " | % of triad |"
            " PyTorch GB/s | ours / PyTorch | OpenBLAS sgemv GB/s | ours / OpenBLAS |"
            + (" IQR µs (ours) | IQR µs (PyTorch) | weight copies |" if detail else "")
            + (" read roof GB/s |" if detail and per_case else "")
            + " core GHz median (min–max) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|---:|" if detail else "")
            + ("---:|" if detail and per_case else "") + "---:|"]
    excluded, unstable = [], []
    for kernel, label in GEMV:
        by_shape: dict[tuple, dict] = {}
        for r in doc["results"]:
            if r["kernel"] == kernel:
                by_shape.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
        for shape in sorted(by_shape, key=lambda s: (s[1] * s[2], s)):
            impls = by_shape[shape]
            o = impls.get(OURS)
            t = next((v for k, v in impls.items() if k.startswith("torch")), None)
            b = next((v for k, v in impls.items() if k.startswith("numpy")), None)
            if o is None:
                continue
            shp = "×".join(str(s) for s in shape)
            if _throttled(o, t, b):
                excluded.append((f"{label} {shp}", o))
                continue
            if _unstable(o, t, b):
                unstable.append((f"{label} {shp}", o))
                continue
            tg = f"{t['gb_per_s']:.1f}" if t else "not measured"
            tr = f"{o['gb_per_s'] / t['gb_per_s']:.2f}×" if t else "—"
            bg = f"{b['gb_per_s']:.1f}" if b else "—"
            br = f"{o['gb_per_s'] / b['gb_per_s']:.2f}×" if b else "—"
            line = (f"| {label} | {shp} | {o['median_s'] * 1e6:.0f} | {o['gb_per_s']:.1f} |"
                    f" {_pct(o.get('pct_stream_read'))} | {_pct(o.get('pct_stream_triad'))} |"
                    f" {tg} | {tr} | {bg} | {br} |")
            if detail:
                ti = f"{t['p25_s'] * 1e6:.0f}–{t['p75_s'] * 1e6:.0f}" if t else "—"
                line += f" {o['p25_s'] * 1e6:.0f}–{o['p75_s'] * 1e6:.0f} | {ti} | {o.get('weight_copies', 1)} |"
                if per_case:
                    line += f" {o['roof_read_gbs']:.1f} |"
            rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded) + _unstable_note(unstable)


def q8_0_table(doc: dict, detail: bool) -> str:
    """Q8_0 decode GEMV (gemv_q8_0): GB/s (34 bytes per 32 weights), % of the
    case's read roof, time against our FP32 / BF16 GEMV and PyTorch's FP32
    F.linear on the dequantized weights (interleaved in the case), and ggml's
    test-backend-ops (its own timing, run after the case)."""
    rows = ["| M×N×K | simd-ml-kernels µs | GB/s | % of read roof | × our FP32 GEMV | × our BF16 GEMV |"
            " PyTorch FP32 µs | ours / PyTorch | ggml µs (own timing) | ggml GB/s | ggml % of roof | ours / ggml |"
            + (" IQR µs (ours) | read roof GB/s | ggml threads (max / timed) |" if detail else "")
            + " core GHz median (min–max) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|---:|" if detail else "") + "---:|"]
    by_shape: dict[tuple, dict] = {}
    for r in doc["results"]:
        if r["kernel"] == "gemv_q8_0":
            by_shape.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
    excluded, unstable = [], []
    for shape in sorted(by_shape, key=lambda s: (s[1] * s[2], s)):
        impls = by_shape[shape]
        o, f, b = impls.get(OURS), impls.get(OURS + " fp32"), impls.get(OURS + " bf16")
        t = next((v for k, v in impls.items() if k.startswith("torch")), None)
        g = impls.get("ggml test-backend-ops")
        if o is None or f is None or b is None:
            continue
        shp = "×".join(str(s) for s in shape)
        if _throttled(o, f, b, t) or (g is not None and "core_ghz" in g and _throttled(g)):
            excluded.append((shp, o))
            continue
        if _unstable(o, f, b, t, g):
            unstable.append((shp, o))
            continue
        us = o["median_s"]
        tt = f"{t['median_s'] * 1e6:.0f}" if t else "not measured"
        tr = f"{t['median_s'] / us:.2f}×" if t else "—"
        if g is not None and "time_s" in g:
            gt, gg, gp, gr = (f"{g['time_s'] * 1e6:.0f}", f"{g['gb_per_s']:.1f}", f"{g['pct_stream_read']:.0f}%",
                              f"{g['time_s'] / us:.2f}×")
        else:
            gt, gg, gp, gr = (g or {}).get("not_measured", "not measured"), "—", "—", "—"
        line = (f"| {shp} | {us * 1e6:.0f} | {o['gb_per_s']:.1f} | {o['pct_stream_read']:.0f}% |"
                f" {f['median_s'] / us:.2f}× | {b['median_s'] / us:.2f}× | {tt} | {tr} | {gt} | {gg} | {gp} | {gr} |")
        if detail:
            th = f"{g['threads_max']} / {g['threads_timed']}" if g is not None and "threads_timed" in g else "—"
            line += f" {o['p25_s'] * 1e6:.0f}–{o['p75_s'] * 1e6:.0f} | {o['roof_read_gbs']:.1f} | {th} |"
        rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded) + _unstable_note(unstable)


def q4_0_table(doc: dict, detail: bool) -> str:
    """Q4_0 decode GEMV (gemv_q4_0): GB/s (18 bytes per 32 weights), % of the
    case's read roof, time against our Q8_0 GEMV (Q8_0 blocks of the same W)
    and FP32 GEMV and PyTorch's FP32 F.linear on the dequantized weights
    (interleaved in the case), and ggml's test-backend-ops (its own timing,
    run after the case)."""
    rows = ["| M×N×K | simd-ml-kernels µs | GB/s | % of read roof | × our Q8_0 GEMV | × our FP32 GEMV |"
            " PyTorch FP32 µs | ours / PyTorch | ggml µs (own timing) | ggml GB/s | ggml % of roof | ours / ggml |"
            + (" IQR µs (ours) | read roof GB/s | ggml threads (max / timed) |" if detail else "")
            + " core GHz median (min–max) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|---:|" if detail else "") + "---:|"]
    by_shape: dict[tuple, dict] = {}
    for r in doc["results"]:
        if r["kernel"] == "gemv_q4_0":
            by_shape.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
    excluded, unstable = [], []
    for shape in sorted(by_shape, key=lambda s: (s[1] * s[2], s)):
        impls = by_shape[shape]
        o, q8, f = impls.get(OURS), impls.get(OURS + " q8_0"), impls.get(OURS + " fp32")
        t = next((v for k, v in impls.items() if k.startswith("torch")), None)
        g = impls.get("ggml test-backend-ops")
        if o is None or q8 is None or f is None:
            continue
        shp = "×".join(str(s) for s in shape)
        if _throttled(o, q8, f, t) or (g is not None and "core_ghz" in g and _throttled(g)):
            excluded.append((shp, o))
            continue
        if _unstable(o, q8, f, t, g):
            unstable.append((shp, o))
            continue
        us = o["median_s"]
        tt = f"{t['median_s'] * 1e6:.0f}" if t else "not measured"
        tr = f"{t['median_s'] / us:.2f}×" if t else "—"
        if g is not None and "time_s" in g:
            gt, gg, gp, gr = (f"{g['time_s'] * 1e6:.0f}", f"{g['gb_per_s']:.1f}", f"{g['pct_stream_read']:.0f}%",
                              f"{g['time_s'] / us:.2f}×")
        else:
            gt, gg, gp, gr = (g or {}).get("not_measured", "not measured"), "—", "—", "—"
        line = (f"| {shp} | {us * 1e6:.0f} | {o['gb_per_s']:.1f} | {o['pct_stream_read']:.0f}% |"
                f" {q8['median_s'] / us:.2f}× | {f['median_s'] / us:.2f}× | {tt} | {tr} | {gt} | {gg} | {gp} | {gr} |")
        if detail:
            th = f"{g['threads_max']} / {g['threads_timed']}" if g is not None and "threads_timed" in g else "—"
            line += f" {o['p25_s'] * 1e6:.0f}–{o['p75_s'] * 1e6:.0f} | {o['roof_read_gbs']:.1f} | {th} |"
        rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded) + _unstable_note(unstable)


def _ggml_source(doc: dict, fmt: str = "q8_0", threads: str = "one thread (OMP_THREAD_LIMIT=1") -> str:
    g = doc["baselines"].get("ggml")
    if not isinstance(g, dict):
        return g or "not measured"
    return (f"llama.cpp {g['tag']} (`{(g['commit'] or '')[:7]}`), `test-backend-ops perf` MUL_MAT {fmt} × f32, "
            f"{threads}, GGML_OPENMP={g['cmake'].get('GGML_OPENMP')}, "
            f"GGML_NATIVE={g['cmake'].get('GGML_NATIVE')}), median of {g['runs_per_case']} runs of its own "
            "mean over ≥ 1 s on the same weights (not interleaved)")


def _roofs(doc: dict) -> str:
    m = doc["machine"]
    def gbs(key):
        v = m.get(key)
        return f"{v:.1f} GB/s" if v is not None else "not measured"
    return (f"single-core read stream {gbs('stream_read_gbs')}, STREAM triad {gbs('stream_triad_gbs')} "
            f"(all-core: read {gbs('stream_read_allcore_gbs')}, triad {gbs('stream_triad_allcore_gbs')})")


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


def _is_threads(doc: dict) -> bool:
    """A threaded decode GEMV run (bench --cases threads): its own tables."""
    return doc["protocol"].get("threads") is not None


def _thr_rows(doc: dict, kernel: str) -> dict:
    """{(shape, threads): {impl: row}} for one kernel of a threads run."""
    out: dict = {}
    for r in doc["results"]:
        if r["kernel"] == kernel and "threads" in r:
            out.setdefault((tuple(r["shape"]), r["threads"]), {})[r["impl"]] = r
    return out


def _ok(*rs: dict | None) -> bool:
    return not any(r is not None and "core_ghz" in r and _throttled(r) for r in rs) and not _unstable(*rs)


def _scaling(o: dict, one: dict | None, th: int) -> str:
    if one is None or not _ok(one):
        return "—"
    return f"{100 * o['gb_per_s'] / (th * one['gb_per_s']):.0f}%"


def _shape_key(shape_th):
    (m, n, k), th = shape_th
    return (n * k, n, k, th)


THREAD_QUANT = [("gemv_q8_0", "Q8_0"), ("gemv_q4_0", "Q4_0")]
THREAD_FLOAT = [("gemv_fp32", "FP32"), ("gemv_fp16", "FP16"), ("gemv_bf16", "BF16")]


def threads_quant_table(doc: dict, detail: bool) -> str:
    rows = ["| format | N×K (Llama-3.2-1B) | threads | simd-ml-kernels µs | GB/s | % of read roof | scaling |"
            " same weights µs | ggml µs (own timing) | ggml % of roof | ours / ggml (same weights) |"
            " ours / ggml (rotated) |"
            + (" read roof GB/s | IQR µs (ours) | ggml threads (max / timed) |" if detail else "")
            + " core GHz median (min–max) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|---:|" if detail else "")
            + "---:|"]
    excluded, unstable = [], []
    for kernel, label in THREAD_QUANT:
        by = _thr_rows(doc, kernel)
        for key in sorted(by, key=_shape_key):
            (m, n, k), th = key
            impls = by[key]
            o, sw, g = impls.get(OURS), impls.get(OURS + " same weights"), impls.get("ggml test-backend-ops")
            if o is None:
                continue
            llama = ", ".join(o.get("llama_3_2_1b", []))
            shp = f"{n}×{k}" + (f" ({llama})" if llama else "")
            name = f"{label} {n}×{k} {th}T"
            if _throttled(o, sw) or (g is not None and "core_ghz" in g and _throttled(g)):
                excluded.append((name, o))
                continue
            if _unstable(o, g):
                unstable.append((name, o))
                continue
            one = by.get(((m, n, k), 1), {}).get(OURS)
            us = o["median_s"]
            swt = f"{sw['median_s'] * 1e6:.0f}" if sw else "—"
            if g is not None and "time_s" in g:
                gt, gp = f"{g['time_s'] * 1e6:.0f}", f"{g['pct_stream_read']:.0f}%"
                gs = f"{g['time_s'] / sw['median_s']:.2f}×" if sw else "—"
                gr = f"{g['time_s'] / us:.2f}×"
            else:
                gt, gp, gs, gr = (g or {}).get("not_measured", "not measured"), "—", "—", "—"
            line = (f"| {label} | {shp} | {th} | {us * 1e6:.0f} | {o['gb_per_s']:.1f} | {o['pct_stream_read']:.0f}% |"
                    f" {_scaling(o, one, th)} | {swt} | {gt} | {gp} | {gs} | {gr} |")
            if detail:
                gth = f"{g['threads_max']} / {g['threads_timed']}" if g is not None and "threads_timed" in g else "—"
                line += f" {o['roof_read_gbs']:.1f} | {o['p25_s'] * 1e6:.0f}–{o['p75_s'] * 1e6:.0f} | {gth} |"
            rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded) + _unstable_note(unstable)


def threads_float_table(doc: dict, detail: bool) -> str:
    rows = ["| weights | N×K | threads | simd-ml-kernels µs | GB/s | % of read roof | scaling |"
            " PyTorch GB/s | ours / PyTorch |" + (" read roof GB/s | IQR µs (ours) |" if detail else "")
            + " core GHz median (min–max) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if detail else "") + "---:|"]
    excluded, unstable = [], []
    for kernel, label in THREAD_FLOAT:
        by = _thr_rows(doc, kernel)
        for key in sorted(by, key=_shape_key):
            (m, n, k), th = key
            impls = by[key]
            o = impls.get(OURS)
            t = next((v for i, v in impls.items() if i.startswith("torch")), None)
            if o is None:
                continue
            name = f"{label} {n}×{k} {th}T"
            if _throttled(o, t):
                excluded.append((name, o))
                continue
            if _unstable(o, t):
                unstable.append((name, o))
                continue
            one = by.get(((m, n, k), 1), {}).get(OURS)
            tg = f"{t['gb_per_s']:.1f}" if t else "—"
            tr = f"{t['median_s'] / o['median_s']:.2f}×" if t else "—"
            line = (f"| {label} | {n}×{k} | {th} | {o['median_s'] * 1e6:.0f} | {o['gb_per_s']:.1f} |"
                    f" {o['pct_stream_read']:.0f}% | {_scaling(o, one, th)} | {tg} | {tr} |")
            if detail:
                line += f" {o['roof_read_gbs']:.1f} | {o['p25_s'] * 1e6:.0f}–{o['p75_s'] * 1e6:.0f} |"
            rows.append(line + f" {_clk(o)} |")
    return "\n".join(rows) + _excluded_note(doc, excluded) + _unstable_note(unstable)


def per_token(doc: dict) -> str:
    """GEMV time of one Llama-3.2-1B decode token: layers × (the shapes of one
    layer × their count) + lm_head, ours (rotated weights) and ggml, per
    format and thread count, from the rows tagged llama_3_2_1b."""
    llama = doc["protocol"]["threads"].get("llama_3_2_1b")
    if not llama:
        return ""
    counts = {"q_proj, o_proj": 2, "k_proj, v_proj": 2, "gate_proj, up_proj": 2, "down_proj": 1}
    lines = []
    for kernel, label in THREAD_QUANT:
        by = _thr_rows(doc, kernel)
        for th in doc["protocol"]["threads"]["counts"]:
            ours = ggml = 0.0
            missing = []
            for key, impls in by.items():
                if key[1] != th:
                    continue
                o, g = impls.get(OURS), impls.get("ggml test-backend-ops")
                for tag in (o or {}).get("llama_3_2_1b", []):
                    mult = llama["layers"] * counts[tag] if tag in counts else 1
                    if not _ok(o) or g is None or "time_s" not in g or not _ok(g):
                        missing.append(tag)
                        continue
                    ours += mult * o["median_s"]
                    ggml += mult * g["time_s"]
            if missing:
                lines.append(f"- {label}, {th} thread{'s' if th > 1 else ''}: not computed "
                             f"({', '.join(sorted(set(missing)))} not shown or not measured)")
            elif ours:
                lines.append(f"- {label}, {th} thread{'s' if th > 1 else ''}: ours {ours * 1e3:.2f} ms, "
                             f"ggml {ggml * 1e3:.2f} ms ({ggml / ours:.2f}×)")
    if not lines:
        return ""
    return ("**Per-token GEMV time, Llama-3.2-1B** (" + llama["source"] + "; "
            f"{llama['layers']} layers × (2 q/o + 2 k/v + 2 gate/up + 1 down) + lm_head, every matrix in the "
            "format shown; ours with rotated weights, ggml on the same weights every run; GEMVs only, not "
            "a tokens/s figure)\n\n" + "\n".join(lines) + "\n")


def threads_section(docs: list[dict], detail: bool, link_prefix: str, latest_only: bool) -> str:
    docs = sorted([d for d in docs if _is_threads(d) and not _is_llama(d)], key=lambda d: d["timestamp_utc"],
                  reverse=True)
    if not docs:
        return "_No committed threaded runs._\n"
    if latest_only:   # newest threaded run per machine label
        newest: dict[str, dict] = {}
        for d in docs:
            newest.setdefault(d["machine"]["label"], d)
        docs = list(newest.values())
    out = []
    for d in docs:
        th = d["protocol"]["threads"]
        cpus = d["machine"].get("pinned_cpus")
        out.append(f"#### {d['machine']['label']}, {d['timestamp_utc'][:10]}, commit `{d['build']['git_commit']}`, "
                   f"{'/'.join(map(str, th['counts']))} threads\n")
        out.append("Source: " + _source(d, link_prefix) + f"; CPUs {cpus}.\n")
        path = d["build"].get("q8_0_path", "not recorded")
        out.append(f"**Q8_0 / Q4_0 decode GEMV at 1–{max(th['counts'])} threads**, path `{path}`. GB/s counts the "
                   "weights, x and y; % of the read roof measured at the same thread count in the same case; "
                   "scaling = GB/s at N threads / (N × GB/s at 1 thread); ours with weights rotated past the "
                   "caches, and on the same weights every call (as ggml runs). ggml: "
                   + _ggml_source(d, "q8_0 or q4_0", "the same thread count (OMP_THREAD_LIMIT = threads")
                   + ".\n")
        out.append(threads_quant_table(d, detail) + "\n")
        out.append(per_token(d))
        out.append("**FP32 / FP16 / BF16 decode GEMV** (x·Wᵀ, W [N, K]), same roof and scaling; PyTorch: "
                   "`F.linear` at `torch.set_num_threads(N)`, timed alone after ours (two OpenMP runtimes would "
                   "spin against each other), FP32 and BF16.\n")
        out.append(threads_float_table(d, detail) + "\n")
    return "\n".join(out).rstrip() + "\n"


def section(docs: list[dict], detail: bool, link_prefix: str, latest_only: bool) -> str:
    """One block per result file, newest first; latest_only keeps the newest
    file per machine label (README), otherwise every run is listed (BENCHMARKS).
    Threaded runs are left to threads_section."""
    docs = [d for d in docs if not _is_threads(d) and not _is_llama(d)]
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
        if any(r["kernel"] in dict(LOWP) for r in d["results"]):
            out.append("**FP16 / BF16-input GEMM, FP32 accumulation and output, single thread** "
                       "(PyTorch: `torch.matmul` on float16 / bfloat16 tensors, which returns "
                       "float16 / bfloat16)\n")
            out.append(lowp_table(d, detail) + "\n")
        if any(r["kernel"] == "sgemm_u8s8" for r in d["results"]):
            path = d["build"].get("u8s8_path", "not recorded")
            out.append(f"**INT8 GEMM (uint8 × per-channel int8, FP32 output), x·Wᵀ, single thread**, "
                       f"path `{path}` (PyTorch: `torch.ops.quantized.linear`, prepacked weights, "
                       "which returns requantized uint8)\n")
            out.append(int8_table(d, detail) + "\n")
        if any(r["kernel"] in dict(GEMV) for r in d["results"]):
            roof = ("read roof: the read stream timed in each case, interleaved with it; STREAM triad of the run: "
                    f"{_roofs(d)}" if d["protocol"].get("gemv_roof") == "per case" else f"bandwidth roofs: {_roofs(d)}")
            out.append("**Decode GEMV y = x·Wᵀ (one row x, W [N, K] as torch.nn.Linear stores it), FP32 output, "
                       f"single thread**; {roof}. GB/s counts the weights, x and y. "
                       "(PyTorch: `F.linear` on the same weights in their dtype, which returns that dtype; "
                       "OpenBLAS: NumPy's matrix-vector product, FP32 only)\n")
            out.append(gemv_table(d, detail) + "\n")
        if any(r["kernel"] == "gemv_q8_0" for r in d["results"]):
            path = d["build"].get("q8_0_path", "not recorded")
            out.append(f"**Q8_0 decode GEMV** (`gemv_q8_0`: ggml block_q8_0 weights, 34 bytes per 32, x quantized "
                       f"to Q8 inside), path `{path}`, single thread; GB/s counts the weights, x and y; × columns: "
                       "time of that GEMV / ours, in the same interleaved case. PyTorch: FP32 `F.linear` on the "
                       "dequantized weights, not a quantized kernel. ggml: " + _ggml_source(d) + ".\n")
            out.append(q8_0_table(d, detail) + "\n")
        if any(r["kernel"] == "gemv_q4_0" for r in d["results"]):
            path = d["build"].get("q8_0_path", "not recorded")
            out.append(f"**Q4_0 decode GEMV** (`gemv_q4_0`: ggml block_q4_0 weights, 18 bytes per 32, x quantized "
                       f"to Q8 inside), path `{path}`, single thread; GB/s counts the weights, x and y; × columns: "
                       "time of that GEMV / ours, in the same interleaved case (our Q8_0 GEMV on the Q8_0 blocks "
                       "of the same W). PyTorch: FP32 `F.linear` on the dequantized weights, not a quantized "
                       "kernel. ggml: " + _ggml_source(d, "q4_0") + ".\n")
            out.append(q4_0_table(d, detail) + "\n")
        out.append("**Activations and normalization, out-of-place, single thread**\n")
        out.append(act_table(d, detail) + "\n")
    return "\n".join(out).rstrip() + "\n"


# ─── Llama-3.2-1B end to end (examples/llama/bench.py) ───────────────────────

LLAMA_W = (("fp32", "F32"), ("q8_0", "Q8_0"), ("q4_0", "Q4_0"))


def _is_llama(doc: dict) -> bool:
    """An end-to-end run (examples/llama/bench.py): its own tables."""
    return "model" in doc and any(r["kernel"].startswith("llama_") for r in doc["results"])


def _llama_rows(doc: dict, kernel: str) -> dict:
    """{(weights, threads, impl, n_depth): row}; n_depth is None for our rows."""
    out: dict = {}
    for r in doc["results"]:
        if r["kernel"] == kernel and "not_measured" not in r:
            out[(r["weights"], r["threads"], r["impl"], r.get("n_depth"))] = r
    return out


def _tps(r: dict | None) -> str:
    if r is None:
        return "not measured"
    mark = " †" if r.get("core_ghz", {}).get("below_min_ghz") else ""
    return f"{r['tokens_per_s']:.1f}{mark}"


def _llama_prompt(doc: dict) -> tuple:
    o = next((r for r in doc["results"] if r["kernel"] == "llama_decode" and r["impl"] == "ours"), {})
    return o.get("prompt_tokens", 0), o.get("gen_tokens", "?")


def llama_speed_table(doc: dict, detail: bool) -> str:
    rows = _llama_rows(doc, "llama_decode")
    depth, _ = _llama_prompt(doc)
    head = ["Weights", "Threads", "Ours tok/s", f"llama.cpp tok/s (depth {depth})", "Ours / llama.cpp"]
    if detail:
        head += ["llama.cpp tok/s (depth 0)", "Ours ms/token", "Ours prefill ms/token", "Clock GHz (min)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for w, label in LLAMA_W:
        for th in sorted({k[1] for k in rows}):
            o = rows.get((w, th, "ours", None))
            if o is None:
                continue
            lc = rows.get((w, th, "llama.cpp", depth))
            cells = [label, str(th), _tps(o), _tps(lc),
                     f"{o['tokens_per_s'] / lc['tokens_per_s']:.2f}×" if lc else "—"]
            if detail:
                c = o["core_ghz"]
                cells += [_tps(rows.get((w, th, "llama.cpp", 0))), f"{o['ms_per_token']:.1f}",
                          f"{o['prefill_ms_per_token']:.1f}", f"{c['median']:.2f} ({c['min']:.2f})"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def llama_breakdown_table(doc: dict) -> str:
    rows = _llama_rows(doc, "llama_decode")
    head = ["Weights", "Threads", "GEMVs", "Attention", "Norms, RoPE, SiLU", "Other (Python/NumPy glue)",
            "Total (profiled)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for w, label in LLAMA_W:
        for th in sorted({k[1] for k in rows}):
            o = rows.get((w, th, "ours", None))
            if o is None:
                continue
            b = o["breakdown_ms_per_token"]
            tot = b["total"]

            def cell(k: str) -> str:
                return f"{b[k]:.2f} ms ({100 * b[k] / tot:.0f}%)"
            lines.append("| " + " | ".join([label, str(th), cell("gemv"), cell("attention"),
                                            cell("norm_rope_silu"), cell("other"), f"{tot:.2f} ms"]) + " |")
    return "\n".join(lines) + "\n"


def llama_ppl_table(doc: dict) -> str:
    rows = _llama_rows(doc, "llama_perplexity")
    if not rows:
        return ""

    def get(w: str, impl: str) -> dict | None:
        return next((r for k, r in rows.items() if k[0] == w and k[2] == impl), None)

    def delta(r: dict | None, f: dict | None) -> str:
        return f"{r['ppl'] - f['ppl']:+.4f}" if r and f and r is not f else "—"

    head = ["Weights", "Ours PPL", "Δ vs our F32", "llama.cpp PPL", "Δ vs llama.cpp F32", "Ours − llama.cpp"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    fo, fg = get("fp32", "ours"), get("fp32", "llama.cpp")
    for w, label in LLAMA_W:
        o, g = get(w, "ours"), get(w, "llama.cpp")
        if o is None and g is None:
            continue
        lines.append("| " + " | ".join([
            label, f"{o['ppl']:.4f}" if o else "not measured", delta(o, fo),
            f"{g['ppl']:.4f} ± {g['ppl_stderr']:.4f}" if g else "not measured", delta(g, fg),
            f"{o['ppl'] - g['ppl']:+.4f}" if o and g else "—"]) + " |")
    return "\n".join(lines) + "\n"


def llama_section(docs: list[dict], detail: bool, link_prefix: str, latest_only: bool) -> str:
    docs = sorted([d for d in docs if _is_llama(d)], key=lambda d: d["timestamp_utc"], reverse=True)
    if not docs:
        return "_Not measured (no committed run of examples/llama/bench.py, which needs SIMD_ML_LLAMA_DIR)._\n"
    if latest_only:
        docs = docs[:1]
    out = []
    for d in docs:
        m, b, lc, p = d["machine"], d["build"], d["baselines"]["llama_cpp"], d["protocol"]
        lcs = (f"llama.cpp {lc.get('tag')} (`{(lc.get('commit') or '')[:12]}`), CPU-only Release, "
               f"GGML_NATIVE={lc['cmake'].get('GGML_NATIVE')}, GGML_OPENMP={lc['cmake'].get('GGML_OPENMP')}, "
               f"GGML_CPU_REPACK={lc['cmake'].get('GGML_CPU_REPACK')}" if isinstance(lc, dict) else lc)
        pt, gen = _llama_prompt(d)
        out.append(f"#### {m['label']}, {d['timestamp_utc'][:10]}, commit `{b['git_commit']}`\n")
        out.append(f"Source: {m['cpu_model']}, {m['core']}, CPUs {m.get('pinned_cpus')}; {d['model']['name']} "
                   "from llama.cpp's own GGUF files (`--pure`), the same bytes on both sides; "
                   f"{lcs}; Python {d['python']}, {b['compiler']}; {d['timestamp_utc']}; "
                   f"[`{d['_file']}`]({link_prefix}{d['_file']}).\n")
        out.append(f"**Tokens/s, greedy decode of {gen} tokens after a {pt}-token prompt.** Ours: the "
                   f"examples/llama/ loop, Python over our GEMVs, median of {p['runs']} runs. llama.cpp: "
                   f"`llama-bench -p 0 -n {gen}`, average, at a depth of {pt} KV entries like ours"
                   + (" (and at depth 0)" if detail else "") + ". × = ours / llama.cpp: above 1, ours is "
                   f"faster. † = clock below {p['min_ghz']} GHz on some thread.\n")
        out.append(llama_speed_table(d, detail) + "\n")
        out.append("**Where our time goes** (one extra profiled run per row; ms per decode token)\n")
        out.append(llama_breakdown_table(d) + "\n")
        ppl = llama_ppl_table(d)
        if ppl:
            t = d["model"]["text"]
            out.append(f"**Perplexity** on `{t['path']}` ({t['tokens']} tokens; llama-perplexity's method: chunks "
                       "of 512, second half scored), ours and `llama-perplexity -c 512` on the same GGUFs\n")
            out.append(ppl + "\n")
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
                      # Absolute: README.md is also the project page on PyPI,
                      # where relative links do not resolve.
                      section(docs, detail=False, latest_only=True,
                              link_prefix="https://github.com/Mattral/simd-ml-kernels/blob/main/")),
        "docs/BENCHMARKS.md": ("results-detail",
                               section(docs, detail=True, link_prefix="../", latest_only=False)),
    }
    threads = {
        "README.md": ("threads-summary",
                      threads_section(docs, detail=False, latest_only=True,
                                      link_prefix="https://github.com/Mattral/simd-ml-kernels/blob/main/")),
        "docs/BENCHMARKS.md": ("threads-detail",
                               threads_section(docs, detail=True, link_prefix="../", latest_only=False)),
    }
    llama = {
        "README.md": ("llama-summary",
                      llama_section(docs, detail=False, latest_only=True,
                                    link_prefix="https://github.com/Mattral/simd-ml-kernels/blob/main/")),
        "docs/BENCHMARKS.md": ("llama-detail",
                               llama_section(docs, detail=True, link_prefix="../", latest_only=False)),
    }
    stale = []
    for rel, (name, body) in targets.items():
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as f:
            old = f.read()
        new = replace_block(old, name, body)
        new = replace_block(new, *threads[rel])
        new = replace_block(new, *llama[rel])
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
