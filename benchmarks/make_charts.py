"""Draw the README charts (SVG, docs/img/) from the committed JSON in
benchmarks/results/. Standard library only; the output depends only on the
JSON, so the same files always give the same bytes (no timestamps).

    python benchmarks/make_charts.py           # rewrite docs/img/*.svg
    python benchmarks/make_charts.py --check   # exit 1 if they are out of date (CI)

Which runs are drawn follows benchmarks/make_tables.py: the newest Llama-3.2-1B
run, and the newest single-thread run per machine label, i.e. the runs whose
tables the README shows. A run must be cited in README.md or the script stops.
Cells the tables leave out (clock below --min-ghz in every attempt, or an
unstable read roof) are left out of the charts by the same tests.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (same directory; shares loading and exclusion rules)

ROOT = mt.ROOT
OUT = os.path.join(ROOT, "docs", "img")
W = 960
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

# Light and dark card colours. The card carries its own background, so the
# chart stays readable whichever GitHub or PyPI theme surrounds it; the dark
# set applies when the viewer's colour scheme is dark. No pure white or black.
STYLE = """
  .bg { fill: #fcfcfb; stroke: #d0d7de; }
  .t1 { fill: #1f2328; }
  .t2 { fill: #57606a; }
  .grid { stroke: #d8dee4; }
  .axis { stroke: #8c959f; }
  .ref { stroke: #57606a; }
  .ours { fill: #2a78d6; }
  .base { fill: #eb6834; }
  .hatch { stroke: #fcfcfb; }
  .lose { fill: #fcfcfb; stroke: #c4432b; }
  .win { fill: #2a78d6; stroke: #fcfcfb; }
  .box { fill: #f0f3f6; stroke: #8c959f; }
  .arrow { stroke: #57606a; fill: none; }
  .head { fill: #57606a; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1d21; stroke: #3d444d; }
    .t1 { fill: #e6edf3; }
    .t2 { fill: #9da7b3; }
    .grid { stroke: #30363d; }
    .axis { stroke: #6e7781; }
    .ref { stroke: #9da7b3; }
    .ours { fill: #3987e5; }
    .base { fill: #d95926; }
    .hatch { stroke: #1a1d21; }
    .lose { fill: #1a1d21; stroke: #f0795a; }
    .win { fill: #3987e5; stroke: #1a1d21; }
    .box { fill: #22272e; stroke: #6e7781; }
    .arrow { stroke: #9da7b3; }
    .head { fill: #9da7b3; }
  }
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def f2(x: float) -> str:
    """Fixed-point coordinates: identical text on every platform."""
    return f"{x:.1f}".rstrip("0").rstrip(".")


def text(x, y, s, cls="t1", size=13, anchor="start", weight="normal") -> str:
    w = f' font-weight="{weight}"' if weight != "normal" else ""
    return (f'<text x="{f2(x)}" y="{f2(y)}" class="{cls}" font-size="{size}" '
            f'text-anchor="{anchor}"{w}>{esc(s)}</text>')


def rect(x, y, w, h, cls, extra="") -> str:
    return f'<rect x="{f2(x)}" y="{f2(y)}" width="{f2(w)}" height="{f2(h)}" class="{cls}"{extra}/>'


def line(x1, y1, x2, y2, cls, extra="") -> str:
    return f'<line x1="{f2(x1)}" y1="{f2(y1)}" x2="{f2(x2)}" y2="{f2(y2)}" class="{cls}"{extra}/>'


def svg(height: float, title: str, desc: str, body: list[str], defs: str = "") -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {f2(height)}" width="{W}" '
        f'height="{f2(height)}" role="img" aria-labelledby="title desc" font-family="{esc(FONT)}">',
        f'<title id="title">{esc(title)}</title>',
        f'<desc id="desc">{esc(desc)}</desc>',
        f"<style>{STYLE}</style>",
        f"<defs>{defs}</defs>" if defs else "",
        rect(0.5, 0.5, W - 1, height - 1, "bg", ' rx="8"'),
        *body,
        "</svg>",
        "",
    ])


def cpu(doc: dict) -> str:
    return doc["machine"]["cpu_model"].replace("(R)", "").replace("(TM)", "").replace("  ", " ")


def source_line(docs: list[dict]) -> list[str]:
    return [f"{cpu(d)} ({d['machine']['label']}), commit {d['build']['git_commit']}, {d['_file']}" for d in docs]


def header(title: str, sources: list[str], y: float = 30) -> tuple[list[str], float]:
    out = [text(24, y, title, size=17, weight="600")]
    for s in sources:
        y += 19
        out.append(text(24, y, s, cls="t2", size=12))
    return out, y


def nice_max(v: float, step: float) -> float:
    n = int(v / step)
    return (n + 1) * step


# ─── Run selection (as make_tables.py's README blocks) ───────────────────────

def cited(docs: list[dict], readme: str) -> list[dict]:
    for d in docs:
        if d["_file"] not in readme:
            raise SystemExit(f"{d['_file']} is not cited in README.md; run benchmarks/make_tables.py first")
    return docs


def newest_llama(docs: list[dict]) -> dict | None:
    ll = sorted([d for d in docs if mt._is_llama(d)], key=lambda d: d["timestamp_utc"], reverse=True)
    return ll[0] if ll else None


def newest_per_label(docs: list[dict]) -> list[dict]:
    ds = sorted([d for d in docs if not mt._is_threads(d) and not mt._is_llama(d)],
                key=lambda d: d["timestamp_utc"], reverse=True)
    seen: set[str] = set()
    return [d for d in ds if not (d["machine"]["label"] in seen or seen.add(d["machine"]["label"]))]


# ─── a. Llama-3.2-1B tokens/s, ours vs llama.cpp ──────────────────────────────

def chart_llama(doc: dict) -> tuple[str, str]:
    rows = mt._llama_rows(doc, "llama_decode")
    depth, gen = mt._llama_prompt(doc)
    min_ghz = doc["protocol"]["min_ghz"]
    data = []
    for w, label in mt.LLAMA_W:
        for th in sorted({k[1] for k in rows}):
            o = rows.get((w, th, "ours", None))
            if o is None:
                continue
            data.append((label, th, o, rows.get((w, th, "llama.cpp", depth))))

    def val(r):
        return None if r is None else r["tokens_per_s"]

    def lab(r):
        return mt._tps(r)   # "12.3" or "12.3 †", as the README table writes it

    any_dagger = any("†" in lab(o) or (lc is not None and "†" in lab(lc)) for _, _, o, lc in data)
    vmax = nice_max(max(v for _, _, o, lc in data for v in (val(o), val(lc)) if v is not None), 10)
    hdr, y = header("Llama-3.2-1B decode, tokens/s: ours vs llama.cpp (higher is better)",
                    source_line([doc]) + [f"Greedy decode of {gen} tokens after a {depth}-token prompt; llama.cpp "
                                          f"{doc['baselines']['llama_cpp'].get('tag')} llama-bench at depth {depth}; "
                                          "same GGUF weights; median of "
                                          f"{doc['protocol']['runs']} runs (ours)."])
    x0, x1 = 190, W - 120
    top = y + 44
    bar, gap, grp = 13, 3, 14
    body = list(hdr)
    # legend: colour and a hatch both tell the two bars apart
    ly = y + 22
    body += [rect(24, ly - 10, 14, 12, "ours"), text(44, ly, "ours (examples/llama/, Python over our GEMVs)", size=12),
             rect(330, ly - 10, 14, 12, "base"), rect(330, ly - 10, 14, 12, "", ' fill="url(#hatch)"'),
             text(350, ly, "llama.cpp", size=12),
             text(W - 24, ly, "ours / llama.cpp", cls="t2", size=12, anchor="end")]
    n = len(data)
    height_plot = n * (2 * bar + gap + grp) + 8
    for i in range(0, int(vmax) + 1, 10):
        x = x0 + (x1 - x0) * i / vmax
        body.append(line(x, top - 6, x, top + height_plot, "grid"))
        body.append(text(x, top + height_plot + 16, str(i), cls="t2", size=11, anchor="middle"))
    body.append(text((x0 + x1) / 2, top + height_plot + 34, "tokens per second", cls="t2", size=12, anchor="middle"))
    yy = top
    prev = None
    for label, th, o, lc in data:
        if prev is not None and prev != label:
            body.append(line(24, yy - grp / 2, W - 24, yy - grp / 2, "grid"))
        prev = label
        body.append(text(24, yy + bar + 4, f"{label}, {th} thread{'s' if th > 1 else ''}", size=13))
        for j, (r, cls) in enumerate(((o, "ours"), (lc, "base"))):
            by = yy + j * (bar + gap)
            if r is None:
                body.append(text(x0 + 4, by + bar - 2, "not measured", cls="t2", size=11))
                continue
            wbar = (x1 - x0) * val(r) / vmax
            body.append(rect(x0, by, wbar, bar, cls, ' rx="2"'))
            if cls == "base":
                body.append(rect(x0, by, wbar, bar, "", ' rx="2" fill="url(#hatch)"'))
            body.append(text(x0 + wbar + 5, by + bar - 2, lab(r), size=11))
        if lc is not None:
            body.append(text(W - 24, yy + bar + 4, f"{val(o) / val(lc):.2f}×", size=13, anchor="end"))
        yy += 2 * bar + gap + grp
    foot = top + height_plot + 58
    body.append(text(24, foot, f"† = clock below {min_ghz} GHz on some thread"
                     + ("" if any_dagger else " (no cell in this run)") + ". Bars start at zero.", cls="t2", size=12))
    defs = ('<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            '<line x1="0" y1="0" x2="0" y2="6" class="hatch" stroke-width="2"/></pattern>')
    parts = [f"{lab_}, {th} thread{'s' if th > 1 else ''}: ours {lab(o)}, llama.cpp {lab(lc) if lc else 'not measured'}"
             + (f" ({val(o) / val(lc):.2f}×)" if lc else "") for lab_, th, o, lc in data]
    desc = (f"Tokens per second, Llama-3.2-1B on {cpu(doc)}, commit {doc['build']['git_commit']}. " + "; ".join(parts) + ".")
    return svg(foot + 20, "Llama-3.2-1B decode tokens/s, ours vs llama.cpp", desc, body, defs), desc


# ─── b. Decode GEMV, % of the measured read roof per weight type ─────────────

def _gemv_cases(doc: dict) -> list[tuple[str, str, float | None, str]]:
    """(weights, shape, % of read roof or None, reason left out), with the
    exclusions of make_tables.gemv_table / q8_0_table / q4_0_table."""
    out = []

    def by_shape(kernel):
        d: dict[tuple, dict] = {}
        for r in doc["results"]:
            if r["kernel"] == kernel:
                d.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
        return d

    def reason(thr, uns):
        if thr:
            return f"not published: clock below {doc['protocol']['min_ghz']} GHz in every attempt"
        return "not published: read roof unstable in this case" if uns else ""

    for kernel, label in mt.GEMV:
        bs = by_shape(kernel)
        for shape in sorted(bs, key=lambda s: (s[1] * s[2], s)):
            im = bs[shape]
            o = im.get(mt.OURS)
            if o is None:
                continue
            t = next((v for k, v in im.items() if k.startswith("torch")), None)
            b = next((v for k, v in im.items() if k.startswith("numpy")), None)
            thr = mt._throttled(o, t, b)
            uns = not thr and mt._unstable(o, t, b)
            out.append((label, "×".join(map(str, shape)), None if thr or uns else o.get("pct_stream_read"),
                        reason(thr, uns)))
    for kernel, label, others in (("gemv_q8_0", "Q8_0", (" fp32", " bf16")), ("gemv_q4_0", "Q4_0", (" q8_0", " fp32"))):
        bs = by_shape(kernel)
        for shape in sorted(bs, key=lambda s: (s[1] * s[2], s)):
            im = bs[shape]
            o = im.get(mt.OURS)
            extra = [im.get(mt.OURS + s) for s in others]
            if o is None or any(e is None for e in extra):
                continue
            t = next((v for k, v in im.items() if k.startswith("torch")), None)
            g = im.get("ggml test-backend-ops")
            thr = mt._throttled(o, *extra, t) or (g is not None and "core_ghz" in g and mt._throttled(g))
            uns = not thr and mt._unstable(o, *extra, t, g)
            out.append((label, "×".join(map(str, shape)), None if thr or uns else o["pct_stream_read"],
                        reason(thr, uns)))
    return out


def chart_gemv(doc: dict) -> tuple[str, str]:
    cases = _gemv_cases(doc)
    per_case = doc["protocol"].get("gemv_roof") == "per case"
    roof = "the read roof timed in the same case" if per_case else "the read stream measured in the same run"
    hdr, y = header("Decode GEMV y = x·Wᵀ: % of the measured read roof, by weight type (higher is better)",
                    source_line([doc]) + [f"Single thread. 100% = {roof}; the GEMV reads every weight once, "
                                          "so memory bandwidth is the limit."])
    x0, x1 = 230, W - 80
    top = y + 26
    bar, gap, grp = 14, 6, 12
    body = list(hdr)
    rows_h = len(cases) * (bar + gap) + grp * (len({c[0] for c in cases}) - 1) + 4
    for p in range(0, 101, 20):
        x = x0 + (x1 - x0) * p / 100
        body.append(line(x, top - 6, x, top + rows_h, "axis" if p == 100 else "grid"))
        body.append(text(x, top + rows_h + 16, f"{p}%", cls="t2", size=11, anchor="middle"))
    yy, prev = top, None
    for label, shp, pct, why in cases:
        if prev is not None and prev != label:
            yy += grp
            body.append(line(24, yy - grp / 2 - gap / 2, W - 24, yy - grp / 2 - gap / 2, "grid"))
        body.append(text(24, yy + bar - 2, f"{label}" if prev != label else "", size=13, weight="600"))
        body.append(text(x0 - 10, yy + bar - 2, shp, cls="t2", size=12, anchor="end"))
        prev = label
        if pct is None:
            body.append(text(x0 + 4, yy + bar - 2, why, cls="t2", size=11))
        else:
            wbar = (x1 - x0) * min(pct, 100) / 100
            body.append(rect(x0, yy, wbar, bar, "ours", ' rx="2"'))
            body.append(text(x0 + wbar + 5, yy + bar - 2, f"{pct:.0f}%", size=11))
        yy += bar + gap
    foot = top + rows_h + 40
    body.append(text(24, foot, "M×N×K per row (one row x, W [N, K]). Bars start at zero. Q8_0 and Q4_0 are ggml's "
                     "block formats (34 and 18 bytes per 32 weights).", cls="t2", size=12))
    groups: dict[str, list[float]] = {}
    for label, _, pct, _ in cases:
        if pct is not None:
            groups.setdefault(label, []).append(pct)
    parts = [f"{k} {min(v):.0f}–{max(v):.0f}%" if len(v) > 1 else f"{k} {v[0]:.0f}%" for k, v in groups.items()]
    left = [f"{label} {shp}" for label, shp, pct, _ in cases if pct is None]
    desc = (f"Decode GEMV as % of the measured read roof on {cpu(doc)}, commit {doc['build']['git_commit']}: "
            + ", ".join(parts) + "." + (f" Not published: {', '.join(left)}." if left else ""))
    return svg(foot + 20, "Decode GEMV, % of the measured read roof", desc, body), desc


# ─── c. Ours / PyTorch or OpenBLAS, every case ───────────────────────────────

ACT_NAMES = {"gelu_exact": "GeLU (erf)", "gelu_tanh": "GeLU (tanh)", "silu": "SiLU", "relu": "ReLU",
             "softmax": "Softmax", "layer_norm": "LayerNorm"}


def _ratio_rows(doc: dict) -> list[tuple[str, list[tuple[str, float]], int]]:
    """(row label, [(case, ours / baseline)], cases left out), with the
    exclusions of the make_tables.py tables. Baselines: OpenBLAS (NumPy) for the
    FP32 GEMM and GEMV, PyTorch CPU otherwise. Q8_0 / Q4_0 are left to the
    llama.cpp chart: their PyTorch row is FP32 on dequantized weights."""
    def by_shape(kernel):
        d: dict[tuple, dict] = {}
        for r in doc["results"]:
            if r["kernel"] == kernel:
                d.setdefault(tuple(r["shape"]), {})[r["impl"]] = r
        return d

    def torch(im):
        return next((v for k, v in im.items() if k.startswith("torch")), None)

    def numpy(im):
        return next((v for k, v in im.items() if k.startswith("numpy")), None)

    rows = []

    def add(label, cases, left):
        if cases or left:
            rows.append((label, cases, left))

    # FP32 GEMM vs OpenBLAS (gemm_table)
    cases, left = [], 0
    for shape, o, b in sorted(mt._pairs(doc, "sgemm")):
        if o is None or b is None:
            continue
        if mt._throttled(o, b):
            left += 1
            continue
        cases.append((f"N={shape[0]}", o["gflops"] / b["gflops"]))
    add("FP32 GEMM vs OpenBLAS", cases, left)
    # FP16 / BF16 GEMM vs PyTorch (lowp_table)
    for kernel, label in mt.LOWP:
        cases, left = [], 0
        bs = by_shape(kernel)
        for shape in sorted(bs, key=lambda s: (s[0] != s[1], s)):
            im = bs[shape]
            o, f, t = im.get(mt.OURS), im.get(mt.OURS + " fp32"), torch(im)
            if o is None or f is None or t is None:
                continue
            if mt._throttled(o, f, t):
                left += 1
                continue
            cases.append(("×".join(map(str, shape)), o["gflops"] / t["gflops"]))
        add(f"{label} GEMM vs PyTorch", cases, left)
    # INT8 GEMM vs PyTorch quantized linear (int8_table)
    cases, left = [], 0
    bs = by_shape("sgemm_u8s8")
    for shape in sorted(bs, key=lambda s: (s[0] != s[1], s)):
        im = bs[shape]
        o, f, cv, t = im.get(mt.OURS), im.get(mt.OURS + " fp32"), im.get(mt.OURS + " fp32_convert"), torch(im)
        if o is None or f is None or cv is None or t is None:
            continue
        if mt._throttled(o, f, cv, t):
            left += 1
            continue
        cases.append(("×".join(map(str, shape)), o["gflops"] / t["gflops"]))
    add("INT8 GEMM vs PyTorch", cases, left)
    # Decode GEMV (gemv_table)
    vs_t: dict[str, list] = {"FP32": [], "FP16/BF16": []}
    vs_b: list = []
    left_t = {"FP32": 0, "FP16/BF16": 0}
    left_b = 0
    for kernel, label in mt.GEMV:
        key = "FP32" if label == "FP32" else "FP16/BF16"
        bs = by_shape(kernel)
        for shape in sorted(bs, key=lambda s: (s[1] * s[2], s)):
            im = bs[shape]
            o, t, b = im.get(mt.OURS), torch(im), numpy(im)
            if o is None:
                continue
            if mt._throttled(o, t, b) or mt._unstable(o, t, b):
                left_t[key] += t is not None
                left_b += b is not None
                continue
            name = f"{label} 1×{'×'.join(map(str, shape[1:]))}"
            if t is not None:
                vs_t[key].append((name, o["gb_per_s"] / t["gb_per_s"]))
            if b is not None:
                vs_b.append((name, o["gb_per_s"] / b["gb_per_s"]))
    add("FP32 decode GEMV vs OpenBLAS", vs_b, left_b)
    add("FP32 decode GEMV vs PyTorch", vs_t["FP32"], left_t["FP32"])
    add("FP16/BF16 decode GEMV vs PyTorch", vs_t["FP16/BF16"], left_t["FP16/BF16"])
    # Activations (act_table)
    for k in mt.ACT_ORDER:
        cases, left = [], 0
        for shape, o, b in mt._pairs(doc, k):
            if o is None or b is None:
                continue
            if mt._throttled(o, b):
                left += 1
                continue
            cases.append(("×".join(map(str, shape)), o["gelem_per_s"] / b["gelem_per_s"]))
        add(f"{ACT_NAMES[k]} vs PyTorch", cases, left)
    return rows


def _rng(vals: list[float]) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:.2f}×" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}–{hi:.2f}×"


def chart_ratio(docs: list[dict]) -> tuple[str, str]:
    panels = [(d, _ratio_rows(d)) for d in docs]
    vmax = nice_max(max(r for _, rows in panels for _, cases, _ in rows for _, r in cases), 0.5)
    hdr, y = header("Speed relative to OpenBLAS or PyTorch, every published case (above 1.0: ours is faster)",
                    source_line(docs) + ["Single thread, the baseline in the same process on the same inputs, "
                                         "interleaved call by call. One mark per case."])
    x0, x1 = 300, W - 150
    rowh = 22
    body = list(hdr)
    ly = y + 22
    body += [f'<circle cx="30" cy="{f2(ly - 4)}" r="5" class="win" stroke-width="1.5"/>',
             text(42, ly, "1.0 or above (ours as fast or faster)", size=12),
             f'<path d="{_diamond(300, ly - 4)}" class="lose" stroke-width="2"/>',
             text(312, ly, "below 1.0 (ours slower)", size=12),
             text(W - 24, ly, "range (cases)", cls="t2", size=12, anchor="end")]
    yy = ly + 18
    descs = []
    for d, rows in panels:
        yy += 14
        body.append(text(24, yy, f"{cpu(d)} ({d['machine']['label']}), commit {d['build']['git_commit']}",
                         size=13, weight="600"))
        yy += 10
        top = yy
        ph = len(rows) * rowh
        steps = int(round(vmax / 0.5))
        for i in range(steps + 1):
            v = i * 0.5
            x = x0 + (x1 - x0) * v / vmax
            body.append(line(x, top, x, top + ph, "grid"))
            body.append(text(x, top + ph + 15, f"{v:g}", cls="t2", size=11, anchor="middle"))
        xr = x0 + (x1 - x0) * 1.0 / vmax
        body.append(line(xr, top - 4, xr, top + ph + 2, "ref", ' stroke-width="1.5" stroke-dasharray="5 3"'))
        below = []
        for label, cases, left in rows:
            cy = yy + rowh / 2
            body.append(text(24, cy + 4, label, size=12))
            body.append(line(x0, cy, x1, cy, "grid", ' stroke-dasharray="1 3"'))
            n = len(cases)
            for j, (cname, r) in enumerate(sorted(cases, key=lambda c: c[1])):
                r = round(r, 2)   # as the tables print it: 0.997 is level (1.00×), not a loss
                x = x0 + (x1 - x0) * r / vmax
                dy = (j - (n - 1) / 2) * (6 / max(n - 1, 1)) if n > 1 else 0
                if r < 1.0:
                    body.append(f'<path d="{_diamond(x, cy + dy)}" class="lose" stroke-width="2"/>')
                    below.append(f"{label.split(' vs ')[0]} {cname} {r:.2f}×")
                else:
                    body.append(f'<circle cx="{f2(x)}" cy="{f2(cy + dy)}" r="5" class="win" stroke-width="1.5"/>')
            rng = (_rng([r for _, r in cases]) + f" ({n})") if cases else "not published"
            if left:
                rng += f", {left} not published"
            body.append(text(W - 24, cy + 4, rng, size=12, anchor="end"))
            yy += rowh
        yy += 30
        rng_all = "; ".join(f"{lab} {_rng([r for _, r in cs])}" for lab, cs, _ in rows if cs)
        descs.append(f"{cpu(d)}, commit {d['build']['git_commit']}: {rng_all}. Below 1.0: "
                     + (", ".join(below) if below else "none") + ".")
    yy += 8
    body.append(text(24, yy, "Axis: ours / baseline, from zero; dashed line = 1.0. \"not published\": the clock fell "
                     "below the P-core threshold in every attempt, as in the README tables.", cls="t2", size=12))
    desc = "Ours divided by the baseline per case. " + " ".join(descs)
    return svg(yy + 20, "Speed relative to OpenBLAS or PyTorch", desc, body), desc


def _diamond(x: float, y: float, r: float = 6) -> str:
    return f"M{f2(x)} {f2(y - r)} L{f2(x + r)} {f2(y)} L{f2(x)} {f2(y + r)} L{f2(x - r)} {f2(y)} Z"


# ─── d. Dispatch (concept, no data) ───────────────────────────────────────────

def chart_dispatch() -> tuple[str, str]:
    body = [text(24, 30, "How a call reaches a kernel body", size=17, weight="600"),
            text(24, 49, "Concept diagram, no measurements. One kernel table is chosen once per process; "
                 "VNNI and BF16-dot are chosen inside a table.", cls="t2", size=12)]
    bw, bh = 182, 92
    ys = 80
    cols = [24, 230, 436]
    stages = [
        ("Python / NumPy", ["sk.sgemm(A, B)", "sk.gemv_q8_0(...)", "float32 / float16 /", "bfloat16 / uint8 arrays"]),
        ("pybind11 binding", ["checks dtype, shape", "and strides;", "releases the GIL"]),
        ("CPUID + XGETBV", ["picks the best table", "the CPU and OS can run;", "SIMD_ML_ISA caps it,", "isa= picks per call"]),
    ]
    for (title, lines_), x in zip(stages, cols):
        body.append(rect(x, ys, bw, bh, "box", ' rx="6" stroke-width="1.2"'))
        body.append(text(x + 12, ys + 22, title, size=14, weight="600"))
        for i, s in enumerate(lines_):
            body.append(text(x + 12, ys + 42 + i * 15, s, cls="t2", size=12))
    for a, b in ((cols[0] + bw, cols[1]), (cols[1] + bw, cols[2])):
        body.append(_arrow(a + 4, ys + bh / 2, b - 4, ys + bh / 2))
    tx, tw = 654, 282
    tables = [
        ("scalar table", "portable C++, every kernel; any CPU"),
        ("AVX2 table", "AVX2 + FMA + F16C; INT8, Q8_0, Q4_0 use", "AVX-VNNI when present, else vpmaddubsw"),
        ("AVX-512 table", "AVX-512F/DQ/BW/VL; INT8, Q8_0, Q4_0 use", "AVX512-VNNI; BF16-dot opt-in"),
    ]
    th = 66
    gap = 14
    ty0 = 62
    for i, (title, *ls) in enumerate(tables):
        ty = ty0 + i * (th + gap)
        body.append(rect(tx, ty, tw, th, "box", ' rx="6" stroke-width="1.2"'))
        body.append(text(tx + 12, ty + 21, title, size=14, weight="600"))
        for j, s in enumerate(ls):
            body.append(text(tx + 12, ty + 39 + j * 14, s, cls="t2", size=11))
        body.append(_arrow(cols[2] + bw + 4, ys + bh / 2, tx - 4, ty + th / 2))
    h = ty0 + 3 * th + 2 * gap + 46
    body.append(text(24, h - 22, "Other architectures (e.g. Apple arm64) build the scalar table only. "
                     "Kernel files: src/kernels/; selection: src/dispatch/dispatch.cpp.", cls="t2", size=12))
    desc = ("A call from Python on NumPy arrays goes through the pybind11 binding, which validates the arrays and "
            "releases the GIL, to the dispatcher, which uses CPUID and XGETBV to choose the scalar, AVX2 or AVX-512 "
            "kernel table; AVX-VNNI, AVX512-VNNI and the opt-in BF16-dot path are chosen inside a table.")
    return svg(h, "Dispatch from Python to a kernel body", desc, body), desc


def _arrow(x1, y1, x2, y2) -> str:
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    hx1, hy1 = x2 + 9 * math.cos(a1), y2 + 9 * math.sin(a1)
    hx2, hy2 = x2 + 9 * math.cos(a2), y2 + 9 * math.sin(a2)
    return (f'<path d="M{f2(x1)} {f2(y1)} L{f2(x2)} {f2(y2)}" class="arrow" stroke-width="1.6"/>'
            f'<path d="M{f2(x2)} {f2(y2)} L{f2(hx1)} {f2(hy1)} L{f2(hx2)} {f2(hy2)} Z" class="head"/>')


# ─── main ─────────────────────────────────────────────────────────────────────

def build() -> dict[str, tuple[str, str]]:
    docs = mt.load()
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    out = {}
    ll = newest_llama(docs)
    if ll is not None:
        cited([ll], readme)
        out["llama_vs_llamacpp.svg"] = chart_llama(ll)
    perf = cited(newest_per_label(docs), readme)
    gemv_docs = [d for d in perf if any(r["kernel"] == "gemv_fp32" for r in d["results"])]
    if gemv_docs:
        out["gemv_roof.svg"] = chart_gemv(gemv_docs[0])
    if perf:
        out["ratio_vs_baseline.svg"] = chart_ratio(perf)
    out["dispatch.svg"] = chart_dispatch()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/img differs from the JSON")
    ap.add_argument("--alt", action="store_true", help="print each chart's description (README alt text)")
    args = ap.parse_args()
    charts = build()
    stale = []
    for name, (content, desc) in charts.items():
        path = os.path.join(OUT, name)
        old = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = f.read()
        if old != content:
            stale.append(f"docs/img/{name}")
            if not args.check:
                os.makedirs(OUT, exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content)
        if args.alt:
            print(f"{name}: {desc}\n")
    if args.check and stale:
        print("out of date (run python benchmarks/make_charts.py):", ", ".join(stale))
        return 1
    print("updated: " + (", ".join(stale) if stale else "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
