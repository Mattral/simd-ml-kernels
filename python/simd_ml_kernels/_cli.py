"""Command-line interface: ``python -m simd_ml_kernels {info,selftest}``.

``info``      what was compiled and what this CPU supports.
``selftest``  every kernel against a NumPy float64 reference with the error
              bounds the test suite asserts; exits 1 if any check fails. This
              is the one command to run to trust an install (< 1 s).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from typing import Callable

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
                                     "isa_compiled", "openmp")},
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
        # 1x1x1, tails of the 8-wide tiles, the small direct path (<= 128)
        # and the packed path (> 128).
        for m, k, n in [(1, 1, 1), (7, 9, 5), (33, 17, 65), (130, 129, 131)]:
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
        ("sgemm_bf16", _gemm_check(lambda a, b: sk.sgemm_bf16(a, b), _as_bf16)),
        ("gelu (exact)", _elementwise_check(sk.gelu, _gelu_erf, atol=2e-6)),
        ("gelu (tanh)", _elementwise_check(lambda x: sk.gelu(x, approximate="tanh"),
                                           _gelu_tanh, atol=2e-6)),
        ("relu", _elementwise_check(sk.relu, lambda x: np.maximum(x, 0.0), atol=1e-30)),
        ("silu", _elementwise_check(sk.silu, lambda x: x / (1.0 + np.exp(-x)), atol=2e-6)),
        ("softmax", _softmax_check),
        ("layer_norm", _layer_norm_check),
    ]
    if sk.f16c_available():
        checks.insert(1, ("sgemm_f16", _gemm_check(lambda a, b: sk.sgemm_f16(a, b), _as_f16)))
    return checks


def cmd_selftest(args: argparse.Namespace) -> int:
    d = info_dict()
    print(f"simd_ml_kernels {d['package']['version']} ({d['build']['git_commit']}), "
          f"{d['cpu']['model']}, runtime ISA {d['cpu']['runtime_isa']}")
    failed = 0
    for name, check in selftest_checks():
        try:
            worst, bound, detail = check()
            ok = worst <= bound
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            worst, ok, detail = float("nan"), False, f"{type(exc).__name__}: {exc}"
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<13} {worst:8.3f}  {detail}")
    if not sk.f16c_available():
        print("  SKIP  sgemm_f16     (CPU has no F16C)")
    print("selftest:", "FAILED" if failed else "passed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simd_ml_kernels")
    sub = parser.add_subparsers(dest="command", required=True)
    p_info = sub.add_parser("info", help="build configuration and CPU features")
    p_info.add_argument("--json", action="store_true", help="print JSON")
    p_info.set_defaults(func=cmd_info)
    p_self = sub.add_parser("selftest", help="check every kernel against float64; exit 1 on failure")
    p_self.set_defaults(func=cmd_selftest)
    args = parser.parse_args(argv)
    return args.func(args)
