"""python -m simd_ml_kernels {info,selftest}."""

import json
import subprocess
import sys

import numpy as np
import pytest

simd_ml_kernels = pytest.importorskip("simd_ml_kernels")
from simd_ml_kernels import _cli  # noqa: E402


def test_selftest_passes_as_a_subprocess():
    r = subprocess.run([sys.executable, "-m", "simd_ml_kernels", "selftest"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "selftest: passed" in r.stdout


def test_selftest_fails_when_a_kernel_is_wrong(monkeypatch, capsys):
    # A ReLU that is off by one ulp-scale amount must be caught: the check
    # compares against float64 with an absolute bound of 1e-30.
    monkeypatch.setattr(_cli.sk, "relu", lambda x: np.maximum(x, 0) + np.float32(1e-6))
    assert _cli.main(["selftest"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  relu" in out and "selftest: FAILED" in out


def test_selftest_reports_exceptions_as_failures(monkeypatch, capsys):
    def boom(x, **kw):
        raise RuntimeError("kernel crashed")
    monkeypatch.setattr(_cli.sk, "silu", boom)
    assert _cli.main(["selftest"]) == 1
    assert "kernel crashed" in capsys.readouterr().out


def test_info_json_has_build_and_cpu_fields(capsys):
    assert _cli.main(["info", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    for key in ("version", "git_commit", "compiler", "kernel_flags", "isa_compiled", "openmp"):
        assert key in d["build"]
    assert d["cpu"]["runtime_isa"] == simd_ml_kernels.detected_isa()
    assert d["package"]["version"] == simd_ml_kernels.__version__
