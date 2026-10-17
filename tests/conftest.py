"""
tests/conftest.py — pytest configuration for the simd-ml-kernels test suite.

The Python tests run against the installed package:

    pip install .          # or: pip install -e . --no-build-isolation

The extension is a module inside the package (simd_ml_kernels._core), so a
bare CMake build directory is not importable on its own; earlier versions of
this file put build/ on sys.path, which also let a stale build shadow the
installed package. The module actually imported is printed in the pytest
header (visible with -v).
"""

from __future__ import annotations


def pytest_report_header(config):
    try:
        import simd_ml_kernels
    except ImportError as exc:
        return f"simd_ml_kernels: not importable ({exc}); run `pip install .` first"
    return f"simd_ml_kernels {simd_ml_kernels.__version__}: {simd_ml_kernels.__file__}"


# ─── Every usable kernel table ────────────────────────────────────────────────
# Each test module runs once per kernel table this CPU can run (scalar, avx2,
# avx512), selected process-wide with the private _core._set_isa(). The table
# is also put in SIMD_ML_ISA, so subprocesses (CLI tests) use the same one.

import os

import pytest


def _usable_isas():
    try:
        import simd_ml_kernels
    except ImportError:
        return ["unavailable"]
    return simd_ml_kernels.build_info()["isa_usable"].split()


@pytest.fixture(autouse=True, scope="module", params=_usable_isas())
def kernel_isa(request):
    if request.param == "unavailable":
        yield None
        return
    import simd_ml_kernels

    previous, previous_env = simd_ml_kernels.detected_isa(), os.environ.get("SIMD_ML_ISA")
    simd_ml_kernels._core._set_isa(request.param)
    os.environ["SIMD_ML_ISA"] = request.param
    yield request.param
    simd_ml_kernels._core._set_isa(previous)
    if previous_env is None:
        os.environ.pop("SIMD_ML_ISA", None)
    else:
        os.environ["SIMD_ML_ISA"] = previous_env
