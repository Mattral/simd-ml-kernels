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
import re
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
                                     "sgemm_blocking", "u8s8_path", "q8_0_path")},
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
        # M <= 8 with trans_b) and the packed path.
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
        # 1x1x1, tile tails, trans_b, zero points 0..255, K past one INT8
        # block (KC = 1536 on a 48 KiB-L1D core), and one row with trans_b
        # (read in place, int8_rows.hpp).
        for m, k, n, tb, zp in [(1, 1, 1, False, 0), (7, 9, 5, False, 128), (3, 70, 33, True, 255),
                                (33, 17, 65, False, 7), (130, 1601, 131, True, 128), (1, 1601, 37, True, 3)]:
            a = rng.integers(0, 256, (m, k), dtype=np.uint8)
            b = rng.integers(-128, 128, (k, n), dtype=np.int8)
            scales = rng.uniform(1e-3, 1e-1, n).astype(np.float32)
            a_scale = 0.02
            c = run(a, np.ascontiguousarray(b.T) if tb else b, a_scale, zp, scales, tb).astype(np.float64)
            # Exact sums in int64 (NumPy's integer matmul does not use BLAS):
            # under Intel SDE -skx, OpenBLAS's float64 matmul of these
            # integers was off by up to 3.4e5 (exact natively and under -hsw).
            ai, bi = a.astype(np.int64) - zp, b.astype(np.int64)
            sc = np.float64(np.float32(a_scale)) * scales.astype(np.float64)
            mag = (np.abs(ai) @ np.abs(bi)).astype(np.float64) * np.abs(sc)
            ratio = np.max(np.abs(c - (ai @ bi).astype(np.float64) * sc) / (2 * (k + 2) * U32 * mag + 1e-300))
            if ratio > worst:
                worst, detail = float(ratio), f"{m}x{k}x{n}{' trans_b' if tb else ''}"
        return worst, 1.0, f"max err / (2*(K+2)*u*|s||A-zp||B|), worst at {detail}"
    return check


def q8_0_blocks(wq: np.ndarray, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Split Q8_0 bytes (quantize_q8_0) of W [n, k] into the float16 scales as
    float64 [n, k/32] and the int8 quants [n, k/32, 32]."""
    b = np.ascontiguousarray(wq).reshape(n, k // 32, 34)
    d = np.ascontiguousarray(b[:, :, :2]).view("<f2")[:, :, 0].astype(np.float64)
    return d, np.ascontiguousarray(b[:, :, 2:]).view(np.int8)


def q8_0_x_scales(x: np.ndarray) -> np.ndarray:
    """The float32 scales gemv_q8_0 uses for x [k]: max|x| / 127 per block."""
    amax = np.abs(x.astype(np.float32)).reshape(-1, 32).max(axis=1)
    return (amax / np.float32(127)).astype(np.float32)


def q8_0_reference(x: np.ndarray, wq: np.ndarray, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """gemv_q8_0's result taken exactly (float64) from the same quantized
    operands, and its bound's magnitude sum_b |dw_b*dx_b| * sum_i |qw*qx|.
    Block sums in int64 (no BLAS: see _u8s8_check)."""
    dw, qw = q8_0_blocks(wq, n, k)
    _, qx = q8_0_blocks(sk.quantize_q8_0(x), 1, k)
    sc = dw * q8_0_x_scales(x).astype(np.float64)[None, :]
    prod = qw.astype(np.int64) * qx.astype(np.int64)
    return ((prod.sum(axis=2).astype(np.float64) * sc).sum(axis=1),
            (np.abs(prod).sum(axis=2).astype(np.float64) * np.abs(sc)).sum(axis=1))


def q4_0_blocks(wq: np.ndarray, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Split Q4_0 bytes (quantize_q4_0) of W [n, k] into the float16 scales as
    float64 [n, k/32] and the signed quants q - 8 as int8 [n, k/32, 32]
    (low nibbles: values 0-15, high nibbles: 16-31)."""
    b = np.ascontiguousarray(wq).reshape(n, k // 32, 18)
    d = np.ascontiguousarray(b[:, :, :2]).view("<f2")[:, :, 0].astype(np.float64)
    qs = b[:, :, 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=2).astype(np.int8) - np.int8(8)
    return d, q


def q4_0_reference(x: np.ndarray, wq: np.ndarray, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """gemv_q4_0's result taken exactly (float64) from the same quantized
    operands (W's Q4_0 blocks, x's Q8 quants and float32 scales), and its
    bound's magnitude sum_b |dw_b*dx_b| * sum_i |(qw - 8)*qx|. Int64 sums."""
    dw, qw = q4_0_blocks(wq, n, k)
    _, qx = q8_0_blocks(sk.quantize_q8_0(x), 1, k)
    sc = dw * q8_0_x_scales(x).astype(np.float64)[None, :]
    prod = qw.astype(np.int64) * qx.astype(np.int64)
    return ((prod.sum(axis=2).astype(np.float64) * sc).sum(axis=1),
            (np.abs(prod).sum(axis=2).astype(np.float64) * np.abs(sc)).sum(axis=1))


def _q4_0_check():
    """gemv_q4_0 against the exact sum of its quantized operands, with the
    documented bound 2*(K/32 + 4)*u*sum_b |dw*dx|*sum |(qw - 8)*qx|."""
    rng = np.random.default_rng(5)
    worst, detail = 0.0, ""
    for n, k in [(1, 32), (7, 64), (9, 96), (17, 320), (33, 4096)]:
        w = rng.standard_normal((n, k)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        wq = sk.quantize_q4_0(w)
        ref, mag = q4_0_reference(x, wq, n, k)
        y = sk.gemv_q4_0(x, wq, n, k).astype(np.float64)
        ratio = np.max(np.abs(y - ref) / (2 * (k // 32 + 4) * U32 * mag + 1e-300))
        if ratio > worst:
            worst, detail = float(ratio), f"N={n} K={k}"
    return worst, 1.0, f"max err / (2*(K/32+4)*u*sum|dw dx||(qw-8) qx|), worst at {detail}"


def _q8_0_check():
    """gemv_q8_0 against the exact sum of its quantized operands, with the
    documented bound 2*(K/32 + 4)*u*sum_b |dw*dx|*sum |qw*qx|."""
    rng = np.random.default_rng(4)
    worst, detail = 0.0, ""
    # One row, the 8-row groups +- 1, and an LLM-sized K.
    for n, k in [(1, 32), (7, 64), (9, 96), (17, 320), (33, 4096)]:
        w = rng.standard_normal((n, k)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        wq = sk.quantize_q8_0(w)
        ref, mag = q8_0_reference(x, wq, n, k)
        y = sk.gemv_q8_0(x, wq, n, k).astype(np.float64)
        ratio = np.max(np.abs(y - ref) / (2 * (k // 32 + 4) * U32 * mag + 1e-300))
        if ratio > worst:
            worst, detail = float(ratio), f"N={n} K={k}"
    return worst, 1.0, f"max err / (2*(K/32+4)*u*sum|dw dx||qw qx|), worst at {detail}"


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
        ("sgemm_bf16 tb", _gemm_check(
            lambda a, b: sk.sgemm_bf16(a, np.ascontiguousarray(b.T), trans_b=True), _as_bf16)),
        ("sgemm_u8s8", _u8s8_check(lambda a, b, s, zp, bs, tb: sk.sgemm_u8s8(a, b, s, zp, bs, trans_b=tb))),
        ("gemv_q8_0", _q8_0_check),
        ("gemv_q4_0", _q4_0_check),
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
    checks.insert(2, ("sgemm_f16 tb", _gemm_check(
        lambda a, b: sk.sgemm_f16(a, np.ascontiguousarray(b.T), trans_b=True), _as_f16)))
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
BENCH_CASES = ("sgemm", "lowp", "int8", "gemv", "q8_0", "q4_0", "act")
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
# Variables that change which kernel runs; recorded with the thread ones.
KERNEL_ENV = ("SIMD_ML_ISA", "SIMD_ML_BF16_DOT")


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


def _record(results, kernel, impl, shape, t, work, unit, clock, nbytes=None, roofs=None):
    """One result row. unit: "flop" (GEMM), "elem" (elementwise) or "byte"
    (STREAM). nbytes: bytes a memory-bound case must move (GEMV: the weights,
    x and y), recorded as gb_per_s and, against roofs (_stream), as
    pct_stream_read (read-only stream, the GEMV's roof) and pct_stream_triad."""
    ghz = clock["median"]
    r = {"kernel": kernel, "impl": impl, "shape": list(shape), **t, "core_ghz": clock}
    rate = work / t["median_s"]
    if unit == "flop":
        r["gflops"] = rate * 1e-9
        r["flop_per_cycle"] = rate / (ghz * 1e9) if ghz == ghz else None
    elif unit == "byte":
        r["gb_per_s"] = rate * 1e-9
        r["byte_per_cycle"] = rate / (ghz * 1e9) if ghz == ghz else None
    else:
        r["gelem_per_s"] = rate * 1e-9
        r["elem_per_cycle"] = rate / (ghz * 1e9) if ghz == ghz else None
    if nbytes is not None:
        r["bytes"] = int(nbytes)
        r["gb_per_s"] = nbytes / t["median_s"] * 1e-9
        for key in ("read", "triad"):
            roof = (roofs or {}).get(key)
            r["pct_stream_" + key] = 100.0 * r["gb_per_s"] / roof if roof else None
    results.append(r)
    return r


def _llc_bytes() -> int:
    """Size of the largest (last-level) data or unified cache of this core,
    from CPUID leaf 4 / 0x8000001D; 0 if unknown (non-x86)."""
    sizes = [int(c["size_kib"]) * 1024 for c in sk._core._cpu_caches() if c["type"] in ("D", "U")]
    return max(sizes, default=0)


# Bandwidth roofs, over float32 arrays each at least 4x the last-level cache
# (STREAM's rule; never a fixed size: one cloud VM reports a 260 MB LLC) and at
# least 64 MiB:
# - read: sum of one array, 4 bytes per element, nothing written. The roof for
#   kernels that only read their big operand (GEMV weights).
# - triad (McCalpin's STREAM): a = b + s*c, 12 bytes per element counted (two
#   loads, one store; STREAM convention, no write-allocate). The stored line's
#   read-for-ownership is real traffic but not counted, so a read-only kernel
#   can exceed 100% of the triad (FP32 GEMV: 107-137% in dev runs).
STREAM_MIN_ARRAY_BYTES = 64 << 20
STREAM_PROTOCOL = ("float32 arrays of max(4x LLC, 64 MiB) each, same warm-up/reps/clock protocol as the "
                   "kernels, single thread on the pinned core: read = sum of one array, 4 bytes per element, "
                   "read by every read-only loop we have (interleaved): 2, 4, 8 and 16 parts side by side, and "
                   "with the AVX2 kernels the GEMV row pattern (rows of 4096 floats, 4 or 8 rows side by side, "
                   "32-byte loads; 8 rows with the next group prefetched); stream_read_gbs is the fastest (the "
                   "roof for read-only kernels such as GEMV: a lower bound on the core's read limit); triad "
                   "a = b + s*c, 12 bytes per element (STREAM convention, no write-allocate; stream_triad_gbs). "
                   "*_allcore_gbs: OpenMP builds with more than one usable CPU only (part sums), else null")


STREAM_READ_STREAMS = (2, 4, 8, 16)
# The GEMV's own pattern (AVX2 kernels only): rows of 16 KiB (an FP32 row at
# K = 4096), 4 or 8 rows side by side with 32-byte loads, and 8 rows with the
# next group prefetched (the Q8_0 GEMV's pattern). Without these the 2-16 part
# sums read about 5% slower than the FP32 GEMV on 1x4096x11008 (DESIGN.md §2.7).
STREAM_READ_ROWS = ((4, False), (8, False), (8, True))
STREAM_ROW_FLOATS = 4096


def _read_roof_fns(buf: np.ndarray) -> list[tuple[str, Callable[[], object]]]:
    """Every read-only loop of the roof, as (name, fn) over buf: the sum read
    as 2-16 parts side by side, and (with the AVX2 kernels) the GEMV row
    patterns. The roof is the fastest, so more variants can only lower a
    kernel's percentage."""
    fns: list[tuple[str, Callable[[], object]]] = [
        (f"{s} parts", functools.partial(sk._core._stream_read, buf, 1, s)) for s in STREAM_READ_STREAMS]
    if "avx2" in sk.build_info()["isa_usable"].split():
        fns += [(f"{r} rows{' + prefetch' if pf else ''}",
                 functools.partial(sk._core._stream_read_rows, buf, r, STREAM_ROW_FLOATS, pf))
                for r, pf in STREAM_READ_ROWS]
    return fns


def _stream(args, results, show) -> dict:
    """Time the read-only stream (read as 2, 4 and 8 parts side by side,
    interleaved; the roof is the fastest, which can only lower a kernel's
    percentage) and the triad, on one thread and (OpenMP builds, more than one
    usable CPU) on all of them. Returns the GB/s as {"read", "triad",
    "read_allcore", "triad_allcore"} (None if not run)."""
    n = max(4 * _llc_bytes(), STREAM_MIN_ARRAY_BYTES) // 4
    a = np.zeros(n, np.float32)
    b = np.full(n, 1.0, np.float32)
    c = np.full(n, 2.0, np.float32)
    threads = len(_pinned_cpus() or [])
    allcore = bool(sk.build_info().get("openmp")) and threads > 1
    roofs: dict = {}
    for th in [1] + ([threads] if allcore else []):
        suffix = "" if th == 1 else "_allcore"
        impl = "simd_ml_kernels" + ("" if th == 1 else f" {th} threads")
        min_ghz = args.min_ghz if th == 1 else None

        fns = _read_roof_fns(b) if th == 1 else [
            (f"{s} parts", functools.partial(sk._core._stream_read, b, th, s)) for s in STREAM_READ_STREAMS]
        stats, clock = _time([f for _, f in fns], args.warmup, args.reps, min_ghz)
        best = 0.0
        for (name, _), t in zip(fns, stats):
            r = _record(results, "stream_read", f"{impl} {name}", (n,), t, 4.0 * n, "byte", clock)
            show(r)
            best = max(best, r["gb_per_s"])
        roofs["read" + suffix] = best

        def triad(th: int = th) -> None:
            sk._core._stream_triad(a, b, c, 3.0, th)
        stats, clock = _time([triad], args.warmup, args.reps, min_ghz)
        r = _record(results, "stream_triad", impl, (n,), stats[0], 12.0 * n, "byte", clock)
        show(r)
        roofs["triad" + suffix] = r["gb_per_s"]
    roofs.setdefault("read_allcore", None)
    roofs.setdefault("triad_allcore", None)
    return roofs


class _Rotor:
    """Index of the next weight copy, shared by every implementation of a
    case, so that consecutive calls (of any library) read different copies.
    With enough copies to exceed twice the last-level cache, no call finds its
    weights in cache, as in a decode step where every layer's weights were
    evicted by the layers before it; with weights already that large, one copy."""

    def __init__(self, copies: int):
        self.copies, self.i = copies, 0

    def __call__(self) -> int:
        self.i = (self.i + 1) % self.copies
        return self.i


def _gemv_impls(fn: Callable, x: np.ndarray, x32: np.ndarray, ws: list, c: np.ndarray, kernel: str,
                torch: Any) -> list[tuple[str, Callable[[], object]]]:
    """The implementations of one GEMV case, all reading the same weight
    buffers ws in the same rotation: ours, NumPy's sgemv (FP32 only) and
    PyTorch's F.linear (tensors that share ws's memory)."""
    rot = _Rotor(len(ws))

    def ours() -> object:
        return fn(x, ws[rot()], C=c, trans_b=True)

    def numpy_sgemv() -> object:
        return np.matmul(x32[0], ws[rot()].T, out=c[0])

    impls: list[tuple[str, Callable[[], object]]] = [("simd_ml_kernels", ours)]
    if kernel == "gemv_fp32":
        impls.append(("numpy/" + str(_blas_info()["name"]) + " sgemv", numpy_sgemv))
    if torch is not None:
        tdtype = {"gemv_fp32": torch.float32, "gemv_fp16": torch.float16, "gemv_bf16": torch.bfloat16}[kernel]
        tws = [torch.from_numpy(w).view(tdtype) for w in ws]   # same buffers, no copy
        xt = torch.from_numpy(x).view(tdtype)

        def torch_linear() -> object:
            return torch.nn.functional.linear(xt, tws[rot()])

        impls.append((f"torch {torch.__version__} {str(tdtype).replace('torch.', '')} F.linear", torch_linear))
    return impls


# Each GEMV case measures its own read roof: the read stream (2, 4 and 8 parts
# side by side) is timed interleaved with the case's implementations, and the
# fastest of the three is that case's roof. A roof measured once per run
# drifted by several percent between runs on the laptop, and on the cloud VM
# single-core bandwidth moved 5.8-11 GB/s between runs (DESIGN.md §2.7). A case
# where any implementation exceeds ROOF_UNSTABLE_PCT of its own roof is marked
# roof_unstable: the roof itself was not measured reliably there, and
# make_tables.py does not publish the case.
ROOF_UNSTABLE_PCT = 105.0


def _gemv_case(results: list, kernel: str, shape: tuple, impls: list, nbytes: dict, roof_buf: np.ndarray,
               triad: float | None, args: argparse.Namespace) -> list[dict]:
    """Time one GEMV case: impls interleaved with the read stream on roof_buf.
    nbytes[impl] = bytes that implementation must move. Returns the rows."""
    m, n, k = shape
    roof_fns = _read_roof_fns(roof_buf)
    stats, clock = _time([f for _, f in impls] + [f for _, f in roof_fns], args.warmup, args.reps, args.min_ghz)
    streams = {s: 4.0 * roof_buf.size / t["median_s"] * 1e-9 for (s, _), t in zip(roof_fns, stats[len(impls):])}
    roof = max(streams.values())
    rows = []
    for (impl, _), t in zip(impls, stats):
        r = _record(results, kernel, impl, shape, t, 2.0 * m * n * k, "flop", clock,
                    nbytes=nbytes[impl], roofs={"read": roof, "triad": triad})
        r["roof_read_gbs"] = roof
        r["roof_read_streams_gbs"] = streams
        rows.append(r)
    return rows


def _mark_roof_unstable(rows: list[dict]) -> None:
    pct = [float(r["pct_stream_read"]) for r in rows if r.get("pct_stream_read") is not None]
    unstable = any(p > ROOF_UNSTABLE_PCT for p in pct)
    for r in rows:
        r["roof_unstable"] = unstable


# ggml baseline: llama.cpp's test-backend-ops, perf mode, one MUL_MAT case
# (type_a q8_0, type_b f32, m = N, n = 1, k = K) given as a --test-file line.
# test-backend-ops has no thread option (it asks for hardware_concurrency()/2
# threads, ignoring the affinity mask), so it runs with OMP_THREAD_LIMIT=1
# (ggml's CPU backend uses OpenMP and reads the team size it gets) and
# taskset on our CPUs. Its threads are read from /proc every 20 ms
# (_watch_threads). It starts 4 short test-data threads per tensor
# (init_tensor_uniform: min(4, hardware_concurrency/2) std::threads, ~10 ms of
# CPU each) before timing, so threads_max is 5 on the laptop; a run counts
# only if, in its last GGML_TIMED_S seconds (inside the >= 1 s timed loop),
# at most 1 thread was seen and the threads other than the main one used at
# most GGML_OTHER_CPU of the main thread's CPU time (the process total minus
# the main thread, which also counts threads that started and ended between
# samples). Over the whole run the init threads use 7-26% of the main
# thread's CPU on these shapes (filling W with random values), all before
# timing, so the CPU check covers the same window. It times its own way: the op
# repeated in one graph for at least 1 s, mean time per run, same weights
# every run; we report the median of GGML_RUNS such means.
GGML_ENV = "SIMD_ML_GGML_TEST_BACKEND_OPS"
GGML_RUNS = 3
GGML_TIMED_S = 1.0
GGML_OTHER_CPU = 0.05
GGML_LINE = re.compile(r"(\d+) runs -\s+([\d.]+) us/run -\s+(\d+) kB/run -\s+(?:\x1b\[[\d;]*m)?\s*([\d.]+) GB/s")


def _ggml_info(binary: str) -> dict:
    """Tool, tag, commit, build settings and llama_print_system_info() of the
    llama.cpp build that holds `binary` (build/bin/test-backend-ops)."""
    import subprocess

    bindir = os.path.dirname(os.path.abspath(binary))
    cache: dict[str, str] = {}
    for d in (os.path.dirname(bindir), bindir):
        path = os.path.join(d, "CMakeCache.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    mm = re.match(r"^([A-Za-z_0-9]+):[A-Z]+=(.*)$", line.rstrip("\n"))
                    if mm:
                        cache[mm.group(1)] = mm.group(2)
            break
    src = cache.get("CMAKE_HOME_DIRECTORY", "")

    def git(*a: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", src, *a], capture_output=True, text=True,
                                  check=True).stdout.strip() or None
        except (OSError, subprocess.CalledProcessError):
            return None
    libllama = os.path.join(bindir, "libllama.so")
    system_info = "not recorded (no libllama.so next to the binary)"
    if os.path.exists(libllama):
        code = ("import ctypes,sys; f=ctypes.CDLL(sys.argv[1]).llama_print_system_info; "
                "f.restype=ctypes.c_char_p; print(f().decode())")
        try:
            system_info = subprocess.run([sys.executable, "-c", code, libllama], capture_output=True,
                                         text=True, check=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            system_info = f"not recorded ({type(exc).__name__})"
    op = None
    try:
        with open(os.path.join(src, "ggml", "include", "ggml.h"), encoding="utf-8") as f:
            body = f.read()
        body = body[body.index("enum ggml_op {"):]
        names = re.findall(r"^\s*(GGML_OP_\w+)", body[:body.index("};")], re.M)
        op = names.index("GGML_OP_MUL_MAT")
    except (OSError, ValueError):
        pass
    return {"tool": "llama.cpp test-backend-ops perf", "binary": os.path.abspath(binary),
            "tag": git("describe", "--tags", "--always"), "commit": git("rev-parse", "HEAD"),
            "cmake": {k: cache.get(k) for k in ("CMAKE_BUILD_TYPE", "GGML_NATIVE", "GGML_OPENMP",
                                                "GGML_LLAMAFILE", "GGML_CPU_REPACK", "CMAKE_CXX_COMPILER",
                                                "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS")},
            "system_info": system_info, "ggml_op_mul_mat": op,
            "env": {"OMP_THREAD_LIMIT": "1"}, "runs_per_case": GGML_RUNS,
            "thread_check": f"in the last {GGML_TIMED_S:g} s before exit (inside the timed loop): threads_timed "
                            f"(most threads seen) must be 1, and threads other than the main one at most "
                            f"{GGML_OTHER_CPU:.0%} of its CPU time (process minus main thread, /proc ticks, "
                            "cpu_ticks.timed_window); threads_max and the whole-run ticks include "
                            "test-backend-ops' test-data init threads, which run before timing",
            "statistic": "test-backend-ops perf: mean time per run over >= 1 s of back-to-back runs of "
                         "the op on the same tensors; time_s = median of runs_per_case such means; not "
                         "interleaved with the other implementations"}


def _watch_threads(p) -> dict:
    """Poll /proc for subprocess p until it exits: thread counts over time and
    CPU clock ticks (utime + stime) of the process, the main thread and each
    thread seen."""
    import time

    def ticks(path: str) -> int:
        with open(path, encoding="ascii") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])
    t0, series = time.monotonic(), []
    per_thread: dict[str, int] = {}
    total = main = 0
    while p.poll() is None:
        try:
            tids = os.listdir(f"/proc/{p.pid}/task")
            for tid in tids:
                try:
                    per_thread[tid] = ticks(f"/proc/{p.pid}/task/{tid}/stat")
                except (OSError, IndexError, ValueError):
                    pass
            total, main = ticks(f"/proc/{p.pid}/stat"), ticks(f"/proc/{p.pid}/task/{p.pid}/stat")
            series.append((time.monotonic() - t0, len(tids), total, main))
        except (OSError, IndexError, ValueError):
            pass
        time.sleep(0.02)
    end = time.monotonic() - t0
    timed = [x for x in series if x[0] >= end - GGML_TIMED_S]
    win = {"main_thread": timed[-1][3] - timed[0][3],
           "other_threads": (timed[-1][2] - timed[-1][3]) - (timed[0][2] - timed[0][3]),
           "seconds": timed[-1][0] - timed[0][0]} if len(timed) > 1 else None
    return {"threads_max": max((x[1] for x in series), default=0),
            "threads_timed": max((x[1] for x in timed), default=0),
            "cpu_ticks": {"process": total, "main_thread": main, "other_threads": total - main,
                          "per_thread": {("main" if tid == str(p.pid) else tid): v for tid, v in per_thread.items()},
                          "timed_window": win},
            "wall_s": end}


# ggml type id and bytes per 32-value block of the weight formats we compare.
GGML_TYPES = {"q8_0": (8, 34), "q4_0": (2, 18)}


def _ggml_case(info: dict, n: int, k: int, args: argparse.Namespace, fmt: str = "q8_0") -> dict:
    """Run test-backend-ops perf on one MUL_MAT [n, k] (fmt: q8_0 or q4_0
    weights) x [k, 1] (f32), GGML_RUNS times, on our CPUs with one thread.
    Returns the measurement (or why not)."""
    import subprocess
    import tempfile

    tid, bb = GGML_TYPES[fmt]
    row = k // 32 * bb
    line = (f"{info['ggml_op_mul_mat']} 0 {n} 1 1 1 0 2 {tid} {k} {n} 1 1 {bb} {row} {row * n} {row * n} "
            f"0 {k} 1 1 1 4 {4 * k} {4 * k} {4 * k} -")
    cpus = _pinned_cpus() or []
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    out: dict = {"test_file_line": line, "env": {"OMP_THREAD_LIMIT": "1"}}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"mul_mat_{fmt}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(line + "\n")
        cmd = ([] if not cpus else ["taskset", "-c", ",".join(map(str, cpus))]) + \
              [info["binary"], "perf", "-b", "CPU", "-o", "MUL_MAT", "--test-file", path]
        out["command"] = "OMP_THREAD_LIMIT=1 " + " ".join(cmd).replace(path, "<test file>")
        for attempt in range(1, 4):
            runs, clock = [], [sk._core._measure_core_ghz(0.005)]
            for _ in range(GGML_RUNS):
                # stdout to a file, not a pipe: nothing to drain while polling.
                with open(os.path.join(tmp, "out.txt"), "w+", encoding="utf-8") as log:
                    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                    watch = _watch_threads(p)
                    p.wait()
                    log.seek(0)
                    text = log.read()
                mm = GGML_LINE.search(text)
                if p.returncode != 0 or not mm:
                    out["not_measured"] = f"not measured (test-backend-ops exit {p.returncode}: {text.strip()[-200:]})"
                    return out
                ct = watch["cpu_ticks"]
                runs.append({"runs": int(mm.group(1)), "us_per_run": float(mm.group(2)),
                             "kb_per_run": int(mm.group(3)), "tool_gb_per_s": float(mm.group(4)), **watch})
                clock.append(sk._core._measure_core_ghz(0.005))
                if watch["threads_timed"] != 1:
                    out["runs"] = runs
                    out["not_measured"] = f"not measured (thread count {watch['threads_timed']} != 1)"
                    return out
                w = ct["timed_window"]
                if w is None or w["other_threads"] > GGML_OTHER_CPU * max(w["main_thread"], 1):
                    out["runs"] = runs
                    out["not_measured"] = ("not measured (no timed window sampled)" if w is None else
                                           f"not measured (other threads used {w['other_threads']} CPU ticks in "
                                           f"the timed window, > {GGML_OTHER_CPU:.0%} of the main thread's "
                                           f"{w['main_thread']})")
                    return out
            cl = np.asarray(clock)
            if args.min_ghz is None or cl.min() >= args.min_ghz:
                break
        out["threads_max"] = max(r["threads_max"] for r in runs)
        out["threads_timed"] = max(r["threads_timed"] for r in runs)
        out["runs"] = runs
        out["time_s"] = float(np.median([r["us_per_run"] for r in runs])) * 1e-6
        out["core_ghz"] = {"median": float(np.median(cl)), "min": float(cl.min()), "max": float(cl.max()),
                           "samples": len(cl), "attempts": attempt,
                           "below_min_ghz": bool(args.min_ghz is not None and cl.min() < args.min_ghz),
                           "note": "sampled by the bench process on the same CPUs before and after each run"}
    return out


# y = W·x for one token: W [N, K] as torch.nn.Linear stores it. M×N×K:
# 4096×4096 (Llama-7B attention projection), 4096×11008 (Llama-7B MLP down
# projection), 2048×8192 (Llama-3.2-1B MLP down projection).
GEMV_SHAPES = [(1, 4096, 4096), (1, 4096, 11008), (1, 2048, 8192)]


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
        if "gflops" in r:
            rate = f"{r['gflops']:8.1f} GFLOP/s ({r['flop_per_cycle'] or float('nan'):.1f}/cycle)"
        elif "gelem_per_s" in r:
            rate = f"{r['gelem_per_s']:8.3f} Gelem/s ({r['elem_per_cycle'] or float('nan'):.3f}/cycle)"
        else:
            rate = ""
        if "gb_per_s" in r:
            pr, pt = r.get("pct_stream_read"), r.get("pct_stream_triad")
            rate += f" {r['gb_per_s']:7.2f} GB/s" + (f" ({pr:.0f}% read, {pt:.0f}% triad)" if pr and pt else "")
        print(f"  {r['kernel']:<16} {r['impl']:<16} {str(tuple(r['shape'])):<16} "
              f"median {r['median_s'] * 1e3:9.3f} ms  {rate}  @ {r['core_ghz']['median']:.2f} GHz "
              f"[{r['core_ghz']['min']:.2f}-{r['core_ghz']['max']:.2f}, attempts {r['core_ghz']['attempts']}]")

    # Bandwidth roofs first: the GEMV rows below are reported as % of them.
    roofs = _stream(args, results, show)

    for n in args.sizes if "sgemm" in args.cases else []:
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
    for m, n, k in [(s, s, s) for s in args.sizes] + [(1, 4096, 4096)] if "lowp" in args.cases else []:
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
    for m, n, k in [(s, s, s) for s in args.sizes] + [(1, 4096, 4096)] if "int8" in args.cases else []:
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

    # Decode GEMV: y = x·Wᵀ for one row x, W [N, K] (torch.nn.Linear layout),
    # FP32 / FP16 / BF16 weights, FP32 output. Bound by reading W, so reported
    # as GB/s (weights + x + y) and % of the single-core read stream (the roof
    # for a read-only kernel) and of the STREAM triad. Baselines
    # on the same buffers: PyTorch's F.linear in the weights' dtype (oneDNN;
    # 16-bit output for 16-bit inputs) and, for FP32, NumPy's matrix-vector
    # product (BLAS sgemv). Weight copies rotate (_Rotor) so no call finds its
    # weights in the last-level cache.
    llc = _llc_bytes()
    roof_buf = np.full(max(4 * llc, STREAM_MIN_ARRAY_BYTES) // 4, 1.0, np.float32)
    for m, n, k in GEMV_SHAPES if "gemv" in args.cases else []:
        w32 = rng.standard_normal((n, k)).astype(np.float32)
        x32 = rng.standard_normal((m, k)).astype(np.float32)
        c = np.empty((m, n), np.float32)
        for kernel, esize in (("gemv_fp32", 4), ("gemv_fp16", 2), ("gemv_bf16", 2)):
            copies = max(1, -(-2 * llc // (n * k * esize)))
            if kernel == "gemv_fp32":
                ws, x = [w32.copy() for _ in range(copies)], x32
            elif kernel == "gemv_fp16":
                ws, x = [w32.astype(np.float16) for _ in range(copies)], x32.astype(np.float16)
            else:
                ws = [(w32.view(np.uint32) >> 16).astype(np.uint16) for _ in range(copies)]
                x = (x32.view(np.uint32) >> 16).astype(np.uint16)
            fns: dict[str, Callable[..., Any]] = {"gemv_fp32": sk.sgemm, "gemv_fp16": sk.sgemm_f16,
                                                  "gemv_bf16": sk.sgemm_bf16}
            impls = _gemv_impls(fns[kernel], x, x32, ws, c, kernel, torch)
            nbytes = n * k * esize + m * k * esize + m * n * 4
            case_rows = _gemv_case(results, kernel, (m, n, k), impls, {i: nbytes for i, _ in impls},
                                   roof_buf, roofs["triad"], args)
            _mark_roof_unstable(case_rows)
            for r in case_rows:
                r["weight_copies"] = copies
                show(r)
            del ws, impls

    # Q8_0 decode GEMV (gemv_q8_0: ggml block_q8_0 weights, 34 bytes per 32;
    # x quantized inside) on the same shapes, interleaved with our FP32 and
    # BF16 GEMV and PyTorch's FP32 F.linear on the dequantized weights (not a
    # quantized kernel: PyTorch CPU has no Q8_0 GEMV), each reading its own
    # weights from DRAM (rotated qcp); then ggml's own Q8_0 MUL_MAT when
    # SIMD_ML_GGML_TEST_BACKEND_OPS points at llama.cpp's test-backend-ops.
    ggml_bin = os.environ.get(GGML_ENV)
    ggml = _ggml_info(ggml_bin) if ggml_bin and ("q8_0" in args.cases or "q4_0" in args.cases) else None
    for m, n, k in GEMV_SHAPES if "q8_0" in args.cases else []:
        w32 = rng.standard_normal((n, k)).astype(np.float32)
        x32 = rng.standard_normal((m, k)).astype(np.float32)
        wq8 = sk.quantize_q8_0(w32)
        dw, qw = q8_0_blocks(wq8, n, k)
        wdq = (qw.astype(np.float32) * dw.astype(np.float32)[:, :, None]).reshape(n, k)
        x1 = np.ascontiguousarray(x32[0])
        # End-to-end error against the unquantized product x·Wᵀ (float64), and
        # the part due to the weights alone (FP32 product on the dequantized
        # weights); the rest is x's 8-bit quantization.
        y_exact = w32.astype(np.float64) @ x1.astype(np.float64)
        y_q = sk.gemv_q8_0(x1, wq8, n, k).astype(np.float64)
        y_w = wdq.astype(np.float64) @ x1.astype(np.float64)

        def _err(yv: np.ndarray) -> dict:
            d = yv - y_exact
            return {"rel_l2": float(np.linalg.norm(d) / np.linalg.norm(y_exact)),
                    "max_abs_over_max_abs": float(np.abs(d).max() / np.abs(y_exact).max())}
        err = {"data": "W and x standard normal (numpy default_rng(0) stream)",
               "gemv_q8_0": _err(y_q), "weights_only": _err(y_w)}
        del w32, dw, qw, y_exact, y_q, y_w
        qbytes = n * k * 34 // 32
        qcopies = max(1, -(-2 * llc // qbytes))
        fcopies = max(1, -(-2 * llc // (n * k * 4)))
        bcopies = max(1, -(-2 * llc // (n * k * 2)))
        wqs = [wq8.copy() for _ in range(qcopies)]
        wfs = [wdq.copy() for _ in range(fcopies)]
        wbs = [(wdq.view(np.uint32) >> 16).astype(np.uint16) for _ in range(bcopies)]
        xb = (x32.view(np.uint32) >> 16).astype(np.uint16)
        c = np.empty((m, n), np.float32)
        rq, rf, rb = _Rotor(qcopies), _Rotor(fcopies), _Rotor(bcopies)
        impls = [("simd_ml_kernels", lambda: sk.gemv_q8_0(x1, wqs[rq()], n, k)),
                 ("simd_ml_kernels fp32", lambda: sk.sgemm(x32, wfs[rf()], C=c, trans_b=True)),
                 ("simd_ml_kernels bf16", lambda: sk.sgemm_bf16(xb, wbs[rb()], C=c, trans_b=True))]
        qnb = {"simd_ml_kernels": qbytes + 4 * k + 4 * n,
                  "simd_ml_kernels fp32": 4 * n * k + 4 * k + 4 * n,
                  "simd_ml_kernels bf16": 2 * n * k + 2 * k + 4 * n}
        if torch is not None:
            tws = [torch.from_numpy(w) for w in wfs]
            xt = torch.from_numpy(x32)
            name = f"torch {torch.__version__} float32 F.linear (dequantized weights)"
            impls.append((name, lambda: torch.nn.functional.linear(xt, tws[rf()])))
            qnb[name] = qnb["simd_ml_kernels fp32"]
        case_rows = _gemv_case(results, "gemv_q8_0", (m, n, k), impls, qnb, roof_buf, roofs["triad"], args)
        qcp = {"simd_ml_kernels": qcopies, "simd_ml_kernels bf16": bcopies}
        for row in case_rows:
            row["weight_copies"] = qcp.get(row["impl"], fcopies)
        case_rows[0]["err_vs_fp32"] = err
        if ggml is not None:
            gres = _ggml_case(ggml, n, k, args)
            row = {"kernel": "gemv_q8_0", "impl": "ggml test-backend-ops", "shape": [m, n, k], **gres}
            if "time_s" in gres:
                row["bytes"] = qnb["simd_ml_kernels"]
                row["gb_per_s"] = row["bytes"] / gres["time_s"] * 1e-9
                row["roof_read_gbs"] = case_rows[0]["roof_read_gbs"]
                row["pct_stream_read"] = 100.0 * row["gb_per_s"] / row["roof_read_gbs"]
                row["pct_stream_triad"] = 100.0 * row["gb_per_s"] / roofs["triad"] if roofs["triad"] else None
            results.append(row)
            case_rows.append(row)
        _mark_roof_unstable(case_rows)
        for row in case_rows:
            if "median_s" in row:
                show(row)
            else:
                print(f"  gemv_q8_0        ggml             {str((m, n, k)):<16} "
                      + (f"{row['time_s'] * 1e3:9.3f} ms (mean, own timing)  {row['gb_per_s']:7.2f} GB/s "
                         f"({row['pct_stream_read']:.0f}% read)" if "time_s" in row else row.get("not_measured", "")))
        del wqs, wfs, wbs, impls

    # Q4_0 decode GEMV (gemv_q4_0: ggml block_q4_0 weights, 18 bytes per 32;
    # x quantized to Q8 inside) on the same shapes, interleaved with our Q8_0
    # GEMV (Q8_0 blocks of the same W), our FP32 GEMV and PyTorch's FP32
    # F.linear on the dequantized Q4_0 weights, each reading its own weights
    # from DRAM (rotated copies); then ggml's own Q4_0 MUL_MAT.
    for m, n, k in GEMV_SHAPES if "q4_0" in args.cases else []:
        w32 = rng.standard_normal((n, k)).astype(np.float32)
        x32 = rng.standard_normal((m, k)).astype(np.float32)
        wq4 = sk.quantize_q4_0(w32)
        wq8 = sk.quantize_q8_0(w32)
        dw, qw = q4_0_blocks(wq4, n, k)
        wdq = (qw.astype(np.float32) * dw.astype(np.float32)[:, :, None]).reshape(n, k)
        x1 = np.ascontiguousarray(x32[0])
        # End-to-end error against x·Wᵀ (float64); weights_only: FP32 product
        # on the dequantized Q4_0 weights; the rest is x's 8-bit quantization.
        y_exact = w32.astype(np.float64) @ x1.astype(np.float64)
        y_q = sk.gemv_q4_0(x1, wq4, n, k).astype(np.float64)
        y_w = wdq.astype(np.float64) @ x1.astype(np.float64)

        def _err4(yv: np.ndarray) -> dict:
            d = yv - y_exact
            return {"rel_l2": float(np.linalg.norm(d) / np.linalg.norm(y_exact)),
                    "max_abs_over_max_abs": float(np.abs(d).max() / np.abs(y_exact).max())}
        err = {"data": "W and x standard normal (numpy default_rng(0) stream)",
               "gemv_q4_0": _err4(y_q), "weights_only": _err4(y_w)}
        del w32, dw, qw, y_exact, y_q, y_w
        q4bytes, q8bytes = n * k * 18 // 32, n * k * 34 // 32
        c4 = max(1, -(-2 * llc // q4bytes))
        c8 = max(1, -(-2 * llc // q8bytes))
        cf = max(1, -(-2 * llc // (n * k * 4)))
        w4s = [wq4.copy() for _ in range(c4)]
        w8s = [wq8.copy() for _ in range(c8)]
        wfs = [wdq.copy() for _ in range(cf)]
        c = np.empty((m, n), np.float32)
        r4, r8, rf = _Rotor(c4), _Rotor(c8), _Rotor(cf)
        impls = [("simd_ml_kernels", lambda: sk.gemv_q4_0(x1, w4s[r4()], n, k)),
                 ("simd_ml_kernels q8_0", lambda: sk.gemv_q8_0(x1, w8s[r8()], n, k)),
                 ("simd_ml_kernels fp32", lambda: sk.sgemm(x32, wfs[rf()], C=c, trans_b=True))]
        qnb = {"simd_ml_kernels": q4bytes + 4 * k + 4 * n,
               "simd_ml_kernels q8_0": q8bytes + 4 * k + 4 * n,
               "simd_ml_kernels fp32": 4 * n * k + 4 * k + 4 * n}
        if torch is not None:
            tws = [torch.from_numpy(w) for w in wfs]
            xt = torch.from_numpy(x32)
            name = f"torch {torch.__version__} float32 F.linear (dequantized weights)"
            impls.append((name, lambda: torch.nn.functional.linear(xt, tws[rf()])))
            qnb[name] = qnb["simd_ml_kernels fp32"]
        case_rows = _gemv_case(results, "gemv_q4_0", (m, n, k), impls, qnb, roof_buf, roofs["triad"], args)
        cps = {"simd_ml_kernels": c4, "simd_ml_kernels q8_0": c8}
        for row in case_rows:
            row["weight_copies"] = cps.get(row["impl"], cf)
        case_rows[0]["err_vs_fp32"] = err
        if ggml is not None:
            gres = _ggml_case(ggml, n, k, args, "q4_0")
            row = {"kernel": "gemv_q4_0", "impl": "ggml test-backend-ops", "shape": [m, n, k], **gres}
            if "time_s" in gres:
                row["bytes"] = qnb["simd_ml_kernels"]
                row["gb_per_s"] = row["bytes"] / gres["time_s"] * 1e-9
                row["roof_read_gbs"] = case_rows[0]["roof_read_gbs"]
                row["pct_stream_read"] = 100.0 * row["gb_per_s"] / row["roof_read_gbs"]
                row["pct_stream_triad"] = 100.0 * row["gb_per_s"] / roofs["triad"] if roofs["triad"] else None
            results.append(row)
            case_rows.append(row)
        _mark_roof_unstable(case_rows)
        for row in case_rows:
            if "median_s" in row:
                show(row)
            else:
                print(f"  gemv_q4_0        ggml             {str((m, n, k)):<16} "
                      + (f"{row['time_s'] * 1e3:9.3f} ms (mean, own timing)  {row['gb_per_s']:7.2f} GB/s "
                         f"({row['pct_stream_read']:.0f}% read)" if "time_s" in row else row.get("not_measured", "")))
        del w4s, w8s, wfs, impls

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
    for name, (fn, shape) in ours.items() if "act" in args.cases else []:
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
                "stream_read_gbs": roofs["read"],
                "stream_triad_gbs": roofs["triad"],
                "stream_read_allcore_gbs": roofs["read_allcore"],
                "stream_triad_allcore_gbs": roofs["triad_allcore"],
            },
            "build": info_dict()["build"],
            "baselines": {"numpy": np.__version__, "blas": _blas_info(),
                          "torch": None if torch is None else torch.__version__,
                          "ggml": ggml if ggml is not None else
                          f"not measured ({GGML_ENV} not set)" if not ggml_bin else "not measured (no q8_0 or q4_0 cases)"},
            "env": {k: os.environ.get(k) for k in THREAD_ENV + KERNEL_ENV},
            "protocol": {"warmup": args.warmup, "reps": args.reps,
                         "statistic": "median; p25/p75 = interquartile range",
                         "interleaved": "implementations of a case alternate call by call",
                         "clock": "core GHz sampled on the same thread (5 ms dependent "
                                  "add-reg-reg chain, 1 cycle/add) before, every 5 rounds and "
                                  "after each case; per-cycle rates use the median",
                         "min_ghz": args.min_ghz,
                         "stream": STREAM_PROTOCOL,
                         "gemv": "y = x·Wᵀ, one row x, W [N, K]; gb_per_s = (weights + x + y bytes) / median "
                                 "time (Q8_0: 34 bytes per 32 weights, Q4_0: 18); per-case roof: the read-stream loops "
                                 "(protocol.stream) timed interleaved with the case's implementations, "
                                 "roof_read_gbs = the fastest; pct_stream_read = gb_per_s / roof_read_gbs; pct_stream_triad = "
                                 "gb_per_s / machine.stream_triad_gbs (once per run); roof_unstable: an "
                                 f"implementation above {ROOF_UNSTABLE_PCT:.0f}% of its case's roof; weights "
                                 "rotate: ceil(2 x LLC / weight bytes) copies (weight_copies, at least 1), the "
                                 "next copy on every call of any implementation, so no call finds its weights "
                                 "in the last-level cache (ggml's test-backend-ops reuses its weights)",
                         "gemv_roof": "per case",
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
    p_bench.add_argument("--cases", type=lambda v: set(v.split(",")), default=set(BENCH_CASES),
                         help="comma-separated subset of " + ",".join(BENCH_CASES) +
                              " (development; published runs use all)")
    p_bench.add_argument("--allow-threads", action="store_true",
                         help="run even if *_NUM_THREADS != 1 (results not publishable)")
    p_bench.set_defaults(func=cmd_bench)
    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]
    return args.func(args)
