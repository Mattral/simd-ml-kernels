"""
tests/conftest.py — pytest configuration for the IntrinsicML test suite.

If `simd_kernels` is importable (pip install), the tests use that module.
Only when it is not installed are build directories added to sys.path, in
this order:

  1. build_ci/   (CMake out-of-source build used locally and in CI)
  2. build/      (alternative build directory name)
  3. Project root (if .so was copied there manually)

Build directories used to take precedence over the installed package, so a
stale extension left in build/ (e.g. after a failed rebuild) was tested
silently instead of the freshly installed one. The module actually imported
is printed in the pytest header.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Repository root is one level above this file (tests/conftest.py → repo root)
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Candidate build directories, in preference order
_SEARCH_DIRS = [
    REPO_ROOT / "build_ci",
    REPO_ROOT / "build",
    REPO_ROOT,          # .so copied to root (common manual step)
]


def _try_add_path(directory: Path) -> bool:
    """Return True if directory contains a simd_kernels .so and was added to sys.path."""
    so_files = list(directory.glob("simd_kernels*.so")) + \
               list(directory.glob("simd_kernels*.pyd"))   # Windows
    if so_files:
        str_dir = str(directory)
        if str_dir not in sys.path:
            sys.path.insert(0, str_dir)
        return True
    return False


if importlib.util.find_spec("simd_kernels") is None:
    for _d in _SEARCH_DIRS:
        if _try_add_path(_d):
            break


def pytest_report_header(config):
    try:
        import simd_kernels
    except ImportError as exc:
        return f"simd_kernels: not importable ({exc})"
    return f"simd_kernels: {simd_kernels.__file__}"
