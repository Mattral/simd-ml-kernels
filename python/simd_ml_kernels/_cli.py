"""Command-line interface: ``python -m simd_ml_kernels {info,selftest,bench}``.

``info``      what was compiled and what this CPU supports.
``selftest``  every kernel against a NumPy float64 reference with the error
              bounds the test suite asserts; exits 1 if any check fails. This
              is the one command to run to trust an install (< 1 s).
``bench``     timings of every kernel and its baseline, optionally as JSON
              with machine metadata (benchmarks/results/, docs/BENCHMARKS.md).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import sys
from typing import Any, Callable

import numpy as np

import simd_ml_kernels as sk

# Unit roundoff of float32.
U32 = 2.0 ** -24


# ─── info ─────────────────────────────────────────────────────────────────────

def cpu_model() -> str:
    """CPU model string from the OS, or platform.processor() as a fallback."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def info_dict() -> dict:
    b = sk.build_info()
    return {
        "package": {"version": sk.__version__, "path": os.path.dirname(sk.__file__)},
        "build": {k: b[k] for k in ("version", "git_commit", "compiler", "kernel_flags",
                                     "isa_compiled", "openmp", "sgemm_tile",
                                     "sgemm_blocking", "u8s8_path")},
        "cpu": {
            "model": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "runtime_isa": b["runtime_isa"],
            "avx512f_dq": sk.avx512_available(),
            "f16c": sk.f16c_available(),
            "avx512_bf16": sk.bf16_avx512bf16_available(),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def cmd_info(args: argparse.Namespace) -> int:
    d = info_dict()
    if args.json:
        print(json.dumps(d, indent=2))
        return 0
    for section, values in d.items():
        if isinstance(values, dict):
            print(f"{section}:")
            for k, v in values.items():
                print(f"  {k:<14} {v}")
        else:
            print(f"{section:<16} {values}")
    return 0


# ─── selftest ─────────────────────────────────────────────────────────────────
# Each check returns (worst, bound, detail): the check passes when worst <= bound.

def _gemm_check(run: Callable, to_input: Callable) -> Callable[[], tuple]:
    """Elementwise forward-error check |C - AB| <= 2*K*u*(|A||B|) on the
    values actually multiplied (after rounding to the input format)."""
    def check():
        rng = np.random.default_rng(0)
        worst, detail = 0.0, ""
        # 1x1x1, tails of the 6x16 tile, the unpacked paths (small products;
        # M <= 4 with trans_b) and the packed path.
        for m, k, n in [(1, 1, 1), (7, 9, 5), (3, 70, 33), (33, 17, 65), (130, 129, 131)]:
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n)).astype(np.float32)
            a_in, a64 = to_input(a)
            b_in, b64 = to_input(b)
            c = run(a_in, b_in).astype(np.float64)
            scale = np.abs(a64) @ np.abs(b64)
            ratio = np.max(np.abs(c - a64 @ b64) / (2 * k * U32 * scale + 1e-300))
            if ratio > worst:
                worst, detail = float(ratio), f"{m}x{k}x{n}"
        return worst, 1.0, f"max err / (2*K*u*|A||B|), worst at {detail}"
    return check


def _u8s8_check(run: Callable) -> Callable[[], tuple]:
    """sgemm_u8s8 against the exact (float64) value, with the documented bound
    |C - C64| <= 2*(K+2)*u*|a_scale*b_scales[j]|*(|A - zp||B|)."""
    def check():
        rng = np.random.default_rng(2)
        worst, detail = 0.0, ""
        # 1x1x1, tile tails, trans_b, zero points 0..255, and K past one INT8
        # block (KC = 1536 on a 48 KiB-L1D core).
        for m, k, n, tb, zp in [(1, 1, 1, False, 0), (7, 9, 5, False, 128), (3, 70, 33, True, 255),
                                (33, 17, 65, False, 7), (130, 1601, 131, True, 128)]:
            a = rng.integers(0, 256, (m, k), dtype=np.uint8)
            b = rng.integers(-128, 128, (k, n), dtype=np.int8)
            scales = rng.uniform(1e-3, 1e-1, n).astype(np.float32)
            a_scale = 0.02
            c = run(a, np.ascontiguousarray(b.T) if tb else b, a_scale, zp, scales, tb).astype(np.float64)
            a64, b64 = a.astype(np.float64) - zp, b.astype(np.float64)
            sc = np.float64(np.float32(a_scale)) * scales.astype(np.float64)
            mag = (np.abs(a64) @ np.abs(b64)) * np.abs(sc)
            ratio = np.max(np.abs(c - (a64 @ b64) * sc) / (2 * (k + 2) * U32 * mag + 1e-300))
            if ratio > worst:
                worst, detail = float(ratio), f"{m}x{k}x{n}{' trans_b' if tb else ''}"
        return worst, 1.0, f"max err / (2*(K+2)*u*|s||A-zp||B|), worst at {detail}"
    return check


def _as_f32(x):
    return x, x.astype(np.float64)


def _as_f16(x):
    h = x.astype(np.float16)
    return h, h.astype(np.float64)


def _as_bf16(x):
    bits = (x.view(np.uint32) >> 16).astype(np.uint16)
    return bits, (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _elementwise_check(fn: Callable, ref: Callable, atol: float, rtol: float = 0.0,
                       lo: float = -10.0, hi: float = 10.0) -> Callable[[], tuple]:
    def check():
        # Dense sweep plus lengths that end in every 8-wide tail position.
        xs = [np.linspace(lo, hi, 20_001).astype(np.float32)]
        rng = np.random.default_rng(1)
        xs += [rng.uniform(lo, hi, n).astype(np.float32) for n in (1, 7, 9, 15)]
        worst = 0.0
        for x in xs:
            r = ref(x.astype(np.float64))
            err = np.abs(fn(x).astype(np.float64) - r)
            worst = max(worst, float(np.max(err / (atol + rtol * np.abs(r)))))
        return worst, 1.0, f"max err / ({atol:g} + {rtol:g}*|ref|) on [{lo:g}, {hi:g}]"
    return check


def _erf(x):
    return np.vectorize(math.erf)(x)


def _gelu_erf(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _softmax_check():
    rng = np.random.default_rng(2)
    worst = 0.0
    for shape in [(1,), (7,), (8,), (3, 9), (4, 1000)]:
        x = (3.0 * rng.standard_normal(shape)).astype(np.float32)
        x64 = x.astype(np.float64)
        e = np.exp(x64 - x64.max(axis=-1, keepdims=True))
        r = e / e.sum(axis=-1, keepdims=True)
        worst = max(worst, float(np.max(np.abs(sk.softmax(x) - r)) / 1e-6))
    return worst, 1.0, "max abs err / 1e-6"


def _layer_norm_check():
    rng = np.random.default_rng(3)
    worst = 0.0
    for n in (1, 7, 9, 768, 1027):
        x = (50.0 + 100.0 * rng.standard_normal((3, n))).astype(np.float32)
        g = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        x64 = x.astype(np.float64)
        xn = (x64 - x64.mean(-1, keepdims=True)) / np.sqrt(x64.var(-1, keepdims=True) + 1e-5)
        r = xn * g + b
        err = np.abs(sk.layer_norm(x, gamma=g, beta=b) - r)
        worst = max(worst, float(np.max(err / (1e-5 + 1e-4 * np.abs(r)))))
    return worst, 1.0, "max err / (1e-5 + 1e-4*|ref|), gamma and beta"


def selftest_checks() -> list[tuple[str, Callable[[], tuple]]]:
    checks = [
        ("sgemm", _gemm_check(lambda a, b: sk.sgemm(a, b), _as_f32)),
        ("sgemm trans_b", _gemm_check(
            lambda a, b: sk.sgemm(a, np.ascontiguousarray(b.T), trans_b=True), _as_f32)),
        ("sgemm trans_a", _gemm_check(
            lambda a, b: sk.sgemm(np.ascontiguousarray(a.T), b, trans_a=True), _as_f32)),
        ("sgemm_bf16", _gemm_check(lambda a, b: sk.sgemm_bf16(a, b), _as_bf16)),
        ("sgemm_u8s8", _u8s8_check(lambda a, b, s, zp, bs, tb: sk.sgemm_u8s8(a, b, s, zp, bs, trans_b=tb))),
        ("gelu (exact)", _elementwise_check(sk.gelu, _gelu_erf, atol=2e-6)),
        ("gelu (tanh)", _elementwise_check(lambda x: sk.gelu(x, approximate="tanh"),
                                           _gelu_tanh, atol=2e-6)),
        ("relu", _elementwise_check(sk.relu, lambda x: np.maximum(x, 0.0), atol=1e-30)),
        ("silu", _elementwise_check(sk.silu, lambda x: x / (1.0 + np.exp(-x)), atol=1e-35,
                                    rtol=1e-6, lo=-200.0, hi=200.0)),
        ("softmax", _softmax_check),
        ("layer_norm", _layer_norm_check),
    ]
    checks.insert(1, ("sgemm_f16", _gemm_check(lambda a, b: sk.sgemm_f16(a, b), _as_f16)))
    return checks


def cmd_selftest(args: argparse.Namespace) -> int:
    d = info_dict()
    usable = sk.build_info()["isa_usable"].split()
    active = sk.detected_isa()
    print(f"simd_ml_kernels {d['package']['version']} ({d['build']['git_commit']}), "
          f"{d['cpu']['model']}, kernel tables {' '.join(usable)} (active: {active})")
    failed = 0
    try:
        # Every table this CPU can run, not only the active one: a scalar or
        # AVX2 path that is never selected here may be the one another CPU uses.
        for isa in usable:
            sk._core._set_isa(isa)
            print(f" [{isa}]")
            for name, check in selftest_checks():
                try:
                    worst, bound, detail = check()
                    ok = worst <= bound
                except Exception as exc:  # noqa: BLE001 - report every failure, keep going
                    worst, ok, detail = float("nan"), False, f"{type(exc).__name__}: {exc}"
                failed += not ok
                print(f"  {'PASS' if ok else 'FAIL'}  {name:<13} {worst:8.3f}  {detail}")
    finally:
        sk._core._set_isa(active)
    print("selftest:", "FAILED" if failed else "passed")
    return 1 if failed else 0


# ─── bench ────────────────────────────────────────────────────────────────────
# Protocol (docs/BENCHMARKS.md): single thread, the process pinned to one core
# by the caller (taskset / `start /affinity`; the core type is recorded via
# --core), >= 5 warm-up and >= 30 timed calls per case, median and
# interquartile range reported. Baselines run in the same process on the same
# inputs: NumPy's BLAS (OpenBLAS) for GEMM, PyTorch CPU for activations when
# torch is importable. Outputs are allocated per call for every library (the
# out-of-place API users call), except GEMM, which writes into a preallocated C.

BENCH_SCHEMA = "simd-ml-kernels-bench/1"
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _pinned_cpus() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    if sys.platform == "win32":
        import ctypes

        proc, sysmask = ctypes.c_size_t(), ctypes.c_size_t()
        k32 = ctypes.windll.kernel32
        if k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(proc),
                                      ctypes.byref(sysmask)):
            return [i for i in range(64) if proc.value >> i & 1]
    return None


def _blas_info() -> dict:
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg["Build Dependencies"]["blas"]
        return {"name": blas.get("name"), "version": blas.get("version")}
    except Exception:  # noqa: BLE001 - older NumPy: no dict mode
        return {"name": "unknown", "version": "unknown"}


def _time_once(fns, warmup, reps):
    import time

    for _ in range(warmup):
        for fn in fns:
            fn()
    clock = [sk._core._measure_core_ghz(0.005)]
    ts = np.empty((len(fns), reps))
    for i in range(reps):
        for j, fn in enumerate(fns):
            t0 = time.perf_counter_ns()
            fn()
            ts[j, i] = (time.perf_counter_ns() - t0) * 1e-9
        if i % 5 == 4:
            clock.append(sk._core._measure_core_ghz(0.005))
    clock.append(sk._core._measure_core_ghz(0.005))
    return ts, clock


def _time(fns: list[Callable[[], object]], warmup: int, reps: int,
          min_ghz: float | None) -> tuple[list[dict], dict]:
    """Time several implementations of the same case, interleaved call by call
    (A, B, A, B, ...), so that clock and thermal drift during the run affects
    all of them alike; on the reference laptop the core clock moved between
    3.1 and 4.0 GHz between consecutive cases timed one after the other.

    The core clock is sampled (5 ms each) before, every 5 timed rounds and
    after; per-cycle rates use the median sample. A single sample after the
    case was not enough: one run reported 3.07 GHz for a case whose GEMM then
    came out at 37.8 flop/cycle, above the 32 flop/cycle an AVX2 core with two
    256-bit FMA pipes can do. With min_ghz set (e.g. just above the E-core
    maximum on a hybrid CPU), a case with any sample below it is repeated, up
    to 3 attempts; the attempt count is recorded."""
    for attempt in range(1, 4):
        ts, clock = _time_once(fns, warmup, reps)
        if min_ghz is None or min(clock) >= min_ghz:
            break
    stats = []
    for row in ts:
        p25, med, p75 = np.percentile(row, [25, 50, 75])
        stats.append({"median_s": float(med), "p25_s": float(p25), "p75_s": float(p75),
                      "min_s": float(row.min()), "reps": reps})
    c = np.asarray(clock)
    return stats, {"median": float(np.median(c)), "min": float(c.min()), "max": float(c.max()),
                   "samples": len(c), "attempts": attempt,
                   "below_min_ghz": bool(min_ghz is not None and c.min() < min_ghz)}


def _record(results, kernel, impl, shape, t, work, unit, clock):
    ghz = clock["median"]
    r = {"kernel": kernel, "impl": impl, "shape": list(shape), **t, "core_ghz": clock}
    rate = work / t["median_s"]
    if unit == "flop":
        r["gflops"] = rate * 1e-9
        r["flop_per_cycle"] = rate / (ghz * 1e9) if ghz == ghz else None
    else:
        r["gelem_per_s"] = rate * 1e-9
        r["elem_per_cycle"] = rate / (ghz * 1e9) if ghz == ghz else None
    results.append(r)
    return r


def _pin_malloc() -> str | None:
    """Fix glibc malloc's thresholds for the bench process, so allocating an
    activation's output (out-of-place, for both libraries) costs the same in
    every case. With glibc's dynamic defaults, the mmap threshold depends on
    which large blocks were freed earlier in the process; after the FP16/BF16
    GEMM cases every 4 MiB output became a fresh mmap and was page-faulted in
    on each call: SiLU on 2^20 floats measured 0.4-0.75 Gelem/s for ours and
    0.6 for PyTorch instead of about 3 and 2.2 (Lunar Lake, WSL2). Blocks below
    32 MiB (glibc's maximum threshold) now come from the heap, which is never
    trimmed, so outputs reuse warm pages. Returns what was set, or None."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        m_trim_threshold, m_top_pad, m_mmap_threshold = -1, -2, -3
        settings = {"M_MMAP_THRESHOLD": (m_mmap_threshold, 32 << 20),
                    "M_TRIM_THRESHOLD": (m_trim_threshold, 1 << 30),
                    "M_TOP_PAD": (m_top_pad, 64 << 20)}
        if all(libc.mallopt(p, v) == 1 for p, v in settings.values()):
            return "glibc mallopt " + ", ".join(f"{k}={v}" for k, (_, v) in settings.items())
    except (OSError, AttributeError):
        pass
    return None


def cmd_bench(args: argparse.Namespace) -> int:
    bad = {k: os.environ.get(k) for k in THREAD_ENV if os.environ.get(k) != "1"}
    if bad and not args.allow_threads:
        print("bench: set " + " ".join(f"{k}=1" for k in bad) + " so the baselines are "
              "single-threaded (or pass --allow-threads).", file=sys.stderr)
        return 2
    torch: Any
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        torch = None

    malloc = _pin_malloc()
    rng = np.random.default_rng(0)
    results: list[dict] = []
    def show(r):
        rate = (f"{r['gflops']:8.1f} GFLOP/s ({r['flop_per_cycle'] or float('nan'):.1f}/cycle)"
                if "gflops" in r else
                f"{r['gelem_per_s']:8.3f} Gelem/s ({r['elem_per_cycle'] or float('nan'):.3f}/cycle)")
        print(f"  {r['kernel']:<16} {r['impl']:<16} {str(tuple(r['shape'])):<16} "
              f"median {r['median_s'] * 1e3:9.3f} ms  {rate}  @ {r['core_ghz']['median']:.2f} GHz "
              f"[{r['core_ghz']['min']:.2f}-{r['core_ghz']['max']:.2f}, attempts {r['core_ghz']['attempts']}]")

    for n in args.sizes:
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)
        c = np.empty((n, n), np.float32)
        impls: list[tuple[str, Callable[[], object]]] = [
            ("simd_ml_kernels", lambda: sk.sgemm(a, b, C=c)),
            ("numpy/" + str(_blas_info()["name"]), lambda: np.matmul(a, b, out=c)),
        ]
        stats, clock = _time([fn for _, fn in impls], args.warmup, args.reps, args.min_ghz)
        for (impl, _), t in zip(impls, stats):
            show(_record(results, "sgemm", impl, (n, n, n), t, 2.0 * n ** 3, "flop", clock))

    # FP16 / BF16 inputs: against our FP32 GEMM on the same values (the
    # conversion's cost) and PyTorch's matmul on float16 / bfloat16 tensors
    # (oneDNN; it returns float16 / bfloat16, ours float32). The square sizes
    # are compute-bound; 1 x 4096 x 4096 (one row times a weight matrix) is
    # bound by reading B, where 2-byte inputs halve the traffic.
    for m, n, k in [(s, s, s) for s in args.sizes] + [(1, 4096, 4096)]:
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        c = np.empty((m, n), np.float32)
        for kernel, lowp_fn, conv, tdtype in (
                ("sgemm_f16", sk.sgemm_f16, _as_f16, "float16"),
                ("sgemm_bf16", sk.sgemm_bf16, _as_bf16, "bfloat16")):
            impls = [("simd_ml_kernels", functools.partial(lowp_fn, conv(a)[0], conv(b)[0], C=c)),
                     ("simd_ml_kernels fp32", functools.partial(sk.sgemm, a, b, C=c))]
            if torch is not None:
                at = torch.from_numpy(a).to(getattr(torch, tdtype))
                bt = torch.from_numpy(b).to(getattr(torch, tdtype))
                impls.append(("torch " + torch.__version__ + " " + tdtype,
                              functools.partial(torch.matmul, at, bt)))
            stats, clock = _time([f for _, f in impls], args.warmup, args.reps, args.min_ghz)
            for (impl, _), t in zip(impls, stats):
                show(_record(results, kernel, impl, (m, n, k), t, 2.0 * m * n * k, "flop", clock))

    # INT8 (uint8 activations x per-channel int8 weights, float32 out) as a
    # linear layer, x [M, K] @ W[N, K].T: ours through the active path
    # (build.u8s8_path), ours through the FP32 conversion path (what CPUs
    # without VNNI run), our FP32 GEMM on the same layout, and PyTorch's
    # static quantized linear (weights prepacked once, quint8 output with its
    # requantization; ours returns float32).
    for m, n, k in [(s, s, s) for s in args.sizes] + [(1, 4096, 4096)]:
        xq = rng.integers(0, 256, (m, k), dtype=np.uint8)
        wq = rng.integers(-127, 128, (n, k), dtype=np.int8)
        ws = rng.uniform(1e-3, 1e-2, n).astype(np.float32)
        xs, zp = 0.02, 128
        xf = rng.standard_normal((m, k)).astype(np.float32)
        wf = rng.standard_normal((n, k)).astype(np.float32)
        c = np.empty((m, n), np.float32)
        impls = [("simd_ml_kernels", functools.partial(sk.sgemm_u8s8, xq, wq, xs, zp, ws, c, trans_b=True)),
                 ("simd_ml_kernels fp32_convert",
                  functools.partial(sk._core._sgemm_u8s8_convert, xq, wq, xs, zp, ws, c, trans_b=True)),
                 ("simd_ml_kernels fp32", functools.partial(sk.sgemm, xf, wf, C=c, trans_b=True))]
        if torch is not None:
            xt = torch.quantize_per_tensor(torch.from_numpy((xq.astype(np.float32) - zp) * xs), xs, zp,
                                           torch.quint8)
            wt = torch.quantize_per_channel(torch.from_numpy(wq.astype(np.float32) * ws[:, None]),
                                            torch.from_numpy(ws.astype(np.float64)),
                                            torch.zeros(n, dtype=torch.int64), 0, torch.qint8)
            packed = torch.ops.quantized.linear_prepack(wt, None)
            impls.append((f"torch {torch.__version__} quantized.linear ({torch.backends.quantized.engine})",
                          functools.partial(torch.ops.quantized.linear, xt, packed, 1.0, 128)))
        stats, clock = _time([f for _, f in impls], args.warmup, args.reps, args.min_ghz)
        for (impl, _), t in zip(impls, stats):
            show(_record(results, "sgemm_u8s8", impl, (m, n, k), t, 2.0 * m * n * k, "flop", clock))

    x = rng.standard_normal(args.elems).astype(np.float32)
    rows = rng.standard_normal((args.elems // 1024, 1024)).astype(np.float32)
    g = rng.standard_normal(1024).astype(np.float32)
    be = rng.standard_normal(1024).astype(np.float32)
    ours = {
        "gelu_exact": (lambda: sk.gelu(x), x.shape),
        "gelu_tanh": (lambda: sk.gelu(x, approximate="tanh"), x.shape),
        "silu": (lambda: sk.silu(x), x.shape),
        "relu": (lambda: sk.relu(x), x.shape),
        "softmax": (lambda: sk.softmax(rows), rows.shape),
        "layer_norm": (lambda: sk.layer_norm(rows, gamma=g, beta=be), rows.shape),
    }
    theirs = {}
    if torch is not None:
        F = torch.nn.functional
        xt, rt = torch.from_numpy(x), torch.from_numpy(rows)
        gt, bt = torch.from_numpy(g), torch.from_numpy(be)
        theirs = {
            "gelu_exact": lambda: F.gelu(xt),
            "gelu_tanh": lambda: F.gelu(xt, approximate="tanh"),
            "silu": lambda: F.silu(xt),
            "relu": lambda: torch.relu(xt),
            "softmax": lambda: torch.softmax(rt, dim=-1),
            "layer_norm": lambda: F.layer_norm(rt, (1024,), gt, bt, 1e-5),
        }
    for name, (fn, shape) in ours.items():
        impls = [("simd_ml_kernels", fn)]
        if name in theirs:
            impls.append(("torch " + torch.__version__, theirs[name]))
        stats, clock = _time([f for _, f in impls], args.warmup, args.reps, args.min_ghz)
        for (impl, _), t in zip(impls, stats):
            show(_record(results, name, impl, shape, t, int(np.prod(shape)), "elem", clock))

    if args.json:
        import datetime

        doc = {
            "schema": BENCH_SCHEMA,
            "machine": {
                "label": args.label,
                "cpu_model": cpu_model(),
                "core": args.core,
                "pinned_cpus": _pinned_cpus(),
                "caches": sk._core._cpu_caches(),
                "logical_cpus": os.cpu_count(),
                "os": platform.platform(),
                "note": args.note,
            },
            "build": info_dict()["build"],
            "baselines": {"numpy": np.__version__, "blas": _blas_info(),
                          "torch": None if torch is None else torch.__version__},
            "env": {k: os.environ.get(k) for k in THREAD_ENV},
            "protocol": {"warmup": args.warmup, "reps": args.reps,
                         "statistic": "median; p25/p75 = interquartile range",
                         "interleaved": "implementations of a case alternate call by call",
                         "clock": "core GHz sampled on the same thread (5 ms dependent "
                                  "add-reg-reg chain, 1 cycle/add) before, every 5 rounds and "
                                  "after each case; per-cycle rates use the median",
                         "min_ghz": args.min_ghz,
                         "malloc": malloc or "not pinned (allocator defaults)"},
            "command": " ".join(["python -m simd_ml_kernels"] + (args.argv or [])),
            "python": platform.python_version(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print(f"wrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simd_ml_kernels")
    sub = parser.add_subparsers(dest="command", required=True)
    p_info = sub.add_parser("info", help="build configuration and CPU features")
    p_info.add_argument("--json", action="store_true", help="print JSON")
    p_info.set_defaults(func=cmd_info)
    p_self = sub.add_parser("selftest", help="check every kernel against float64; exit 1 on failure")
    p_self.set_defaults(func=cmd_selftest)
    p_bench = sub.add_parser("bench", help="time kernels and baselines; see docs/BENCHMARKS.md")
    p_bench.add_argument("--sizes", type=lambda v: [int(t) for t in v.split(",")],
                         default=[128, 256, 512, 1024, 2048], help="square GEMM sizes")
    p_bench.add_argument("--elems", type=int, default=1 << 20,
                         help="elements per activation call (multiple of 1024)")
    p_bench.add_argument("--warmup", type=int, default=5)
    p_bench.add_argument("--reps", type=int, default=30)
    p_bench.add_argument("--json", metavar="PATH", help="write results with machine metadata")
    p_bench.add_argument("--label", default="unlabelled", help="machine label for tables")
    p_bench.add_argument("--core", default="not recorded",
                         help="core the process is pinned to, e.g. 'P-core (Lion Cove), CPU 2'")
    p_bench.add_argument("--note", default="", help="free text, e.g. power settings")
    p_bench.add_argument("--min-ghz", type=float, default=None,
                         help="repeat a case (up to 3 attempts) if any clock sample is below "
                              "this; on hybrid CPUs set it just above the E-core maximum")
    p_bench.add_argument("--allow-threads", action="store_true",
                         help="run even if *_NUM_THREADS != 1 (results not publishable)")
    p_bench.set_defaults(func=cmd_bench)
    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]
    return args.func(args)
