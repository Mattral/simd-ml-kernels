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
