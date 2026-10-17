"""_core.pyi parameters match the compiled module.

mypy.stubtest (run in CI) finds names missing from, or extra in, the stub, but
it cannot compare signatures here: pybind11 functions have no
__text_signature__, so inspect.signature() fails for them. pybind11 does put
the signature on the first docstring line, e.g.
``layer_norm(x: numpy.ndarray, gamma: object = None, ...) -> ...``; this test
parses that line and compares parameter names, order and default values with
the stub.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import simd_ml_kernels
from simd_ml_kernels import _core

# Next to __init__.py, not _core.so: an editable install keeps the .so in the build tree.
STUB = Path(simd_ml_kernels.__file__).with_name("_core.pyi")


def _params(fn: ast.FunctionDef) -> list[tuple[str, object]]:
    args = fn.args.args
    defaults = [None] * (len(args) - len(fn.args.defaults)) + fn.args.defaults
    out = []
    for a, d in zip(args, defaults):
        if a.arg == "self":
            continue
        out.append((a.arg, "<required>" if d is None else ast.literal_eval(d)))
    return out


def _stub_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in ("__init__", "__call__"):
                    out[f"{node.name}.{item.name}"] = item
    return out


def _runtime_signature(qualname: str) -> ast.FunctionDef:
    obj = _core
    for part in qualname.split("."):
        obj = getattr(obj, part)
    first = obj.__doc__.strip().splitlines()[0]
    # "name(a: T, b: U = 1.0) -> R" -> parse as a def; the return annotation is dropped
    # because pybind11's type names (e.g. numpy.typing.NDArray[numpy.float32]) need not
    # match the stub's spelling.
    head = first.split(") ->")[0] + ")"
    return ast.parse(f"def {head}: ...").body[0]


def _same_default(stub: object, runtime: object) -> bool:
    if isinstance(stub, float) or isinstance(runtime, float):
        # C++ float defaults print as their float32 value (1e-5 -> 9.99999974e-06).
        return np.float32(stub) == np.float32(runtime)
    return stub == runtime


@pytest.mark.parametrize("name", sorted(_stub_functions()))
def test_stub_parameters_match_runtime(name):
    stub = _params(_stub_functions()[name])
    runtime = _params(_runtime_signature(name))
    assert [n for n, _ in stub] == [n for n, _ in runtime], f"{name}: parameter names differ"
    for (pname, s), (_, r) in zip(stub, runtime):
        assert _same_default(s, r), f"{name}({pname}=...): stub default {s!r}, runtime {r!r}"

