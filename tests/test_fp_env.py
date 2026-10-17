"""Importing or running the kernels must not change the floating-point
environment of the host process.

The AVX2/AVX-512 units are compiled with -ffast-math. Linked into an
executable, -ffast-math pulls in crtfastmath.o, which sets FTZ (flush
subnormal results to zero) and DAZ (treat subnormal inputs as zero) in MXCSR
at start-up, changing the results of every float operation in the process.
For a shared object such as the extension it must not. FTZ/DAZ are read here
through their effect on NumPy float32 arithmetic in a fresh interpreter,
before and after import and after calling every kernel on every usable
kernel table.
"""

import os
import subprocess
import sys

PROBE = r"""
import numpy as np

def fp_env():
    tiny = np.float32(1e-40)                                  # subnormal
    daz = bool(tiny * np.float32(1.0) == 0)                   # input flushed
    ftz = bool(np.float32(1e-38) * np.float32(1e-3) == 0)     # result 1e-41 flushed
    return daz, ftz

before = fp_env()
import simd_ml_kernels as sk
after_import = fp_env()
x = np.linspace(-5, 5, 1027).astype(np.float32)
a = np.ones((130, 130), np.float32)
for isa in sk.build_info()["isa_usable"].split():
    sk._core._set_isa(isa)
    sk.sgemm(a, a); sk.sgemm(a[:8, :8].copy(), a[:8, :8].copy())
    sk.sgemm_f16(a.astype(np.float16), a.astype(np.float16))
    sk.sgemm_bf16((a.view(np.uint32) >> 16).astype(np.uint16), (a.view(np.uint32) >> 16).astype(np.uint16))
    sk.gelu(x); sk.gelu(x, approximate="tanh"); sk.relu(x); sk.silu(x)
    sk.softmax(x.reshape(13, 79)); sk.layer_norm(x.reshape(13, 79))
after_calls = fp_env()
print(before, after_import, after_calls)
assert before == (False, False), before
assert after_import == before, after_import
assert after_calls == before, after_calls
"""


def test_import_and_kernels_leave_ftz_daz_unchanged():
    env = {k: v for k, v in os.environ.items() if k != "SIMD_ML_ISA"}
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
