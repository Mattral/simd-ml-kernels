"""
tests/test_precision.py — Numerical Precision Tests vs NumPy/SciPy/PyTorch

Validates that all SIMD kernels agree with reference implementations to
within acceptable FP32 rounding tolerance.

Run with:
    pytest tests/test_precision.py -v

Requirements:
    pip install numpy pytest scipy  (torch is optional)
    pip install -e .  # or: cmake --build build && cp build/simd_ml_kernels*.so .
"""

import math
import os

import numpy as np
import pytest

try:
    import simd_ml_kernels
    from simd_ml_kernels._cli import q4_0_blocks, q4_0_reference, q8_0_blocks, q8_0_reference, q8_0_x_scales
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    pytest.skip("simd_ml_kernels not built — run cmake/pip install first",
                allow_module_level=True)

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.special import erf as scipy_erf
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


RNG = np.random.default_rng(seed=42)

# ─── Reference implementations ───────────────────────────────────────────────

def rand_matrix(rows, cols, dtype=np.float32):
    return RNG.standard_normal((rows, cols)).astype(dtype)

def rand_array(n, dtype=np.float32):
    return RNG.standard_normal(n).astype(dtype)

def gelu_tanh_numpy(x: np.ndarray) -> np.ndarray:
    """Reference GeLU using tanh approximation (matches SIMD kernel formula)."""
    x64 = x.astype(np.float64)
    c = np.sqrt(2.0 / np.pi)
    return (0.5 * x64 * (1.0 + np.tanh(c * (x64 + 0.044715 * x64**3)))).astype(np.float64)

def relu_numpy(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x.astype(np.float64))

def silu_numpy(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    return x64 / (1.0 + np.exp(-x64))

def softmax_numpy(x: np.ndarray, axis=-1) -> np.ndarray:
    x64 = x.astype(np.float64)
    x64 -= x64.max(axis=axis, keepdims=True)
    e = np.exp(x64)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm_numpy(x: np.ndarray, gamma=None, beta=None, eps=1e-5) -> np.ndarray:
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var  = x64.var(axis=-1, keepdims=True)
    xn   = (x64 - mean) / np.sqrt(var + eps)
    if gamma is not None:
        xn = xn * gamma.astype(np.float64)
    if beta is not None:
        xn = xn + beta.astype(np.float64)
    return xn


def assert_close(actual, ref, rtol: float, atol: float, label: str = "") -> None:
    """
    Assert |actual - ref| <= atol + rtol * |ref| (the same combined criterion
    np.allclose uses internally), with a diagnostic message pinpointing the
    worst violation.

    Why not a naive relative error `|actual-ref| / (|ref|+eps)`?
    ---------------------------------------------------------------
    That pattern is numerically unstable near zero-crossings. Functions like
    GeLU, SiLU, and LayerNorm legitimately produce outputs arbitrarily close
    to zero for some inputs. When |ref| is tiny, even a few ULPs of
    disagreement between two independently-implemented approximations (e.g.
    our Cody-Waite tanh vs PyTorch's own tanh backend) inflates the "relative
    error" arbitrarily — the metric is measuring noise in the denominator,
    not a real discrepancy in the kernel.

    This was not a hypothetical concern: test_gelu_vs_pytorch_tanh_approx
    failed intermittently in CI with max_rel_err=1.46e-4 against a 1e-4
    threshold. Root-caused to a single sample x=8.4e-5 (drawn from
    N(0,1) with 4096 samples — expected to occur in most runs) where
    ref=GeLU(x)=4.2e-5 and the actual absolute difference between our kernel
    and PyTorch's was 6e-9 — under one float32 ULP at that scale. The old
    metric reported this as a 1.46e-4 "relative error" purely because the
    denominator was tiny. The combined atol+rtol criterion used here is
    exactly what np.allclose and virtually all numerical test suites use,
    and does not have this failure mode.
    """
    actual64 = np.asarray(actual, dtype=np.float64)
    ref64    = np.asarray(ref,    dtype=np.float64)
    diff     = np.abs(actual64 - ref64)
    bound    = atol + rtol * np.abs(ref64)
    violation = diff - bound
    worst = int(np.argmax(violation))
    max_violation = float(violation.flat[worst])
    assert max_violation <= 0.0, (
        f"{label}: worst violation={max_violation:.3e} "
        f"(|diff|={diff.flat[worst]:.3e}, ref={ref64.flat[worst]:.3e}, "
        f"bound=atol+rtol*|ref|={atol:.1e}+{rtol:.1e}*|ref|={bound.flat[worst]:.3e})"
    )


# ─── GEMM Tests ──────────────────────────────────────────────────────────────

class TestGEMM:
    """Verify simd_ml_kernels.sgemm matches np.dot to within FP32 tolerance."""

    GEMM_TOL = 5e-2  # -ffast-math allows FP reassociation; O(K*eps_mach) accumulated error
    GEMM_ATOL = 1e-4  # absolute floor: GEMM output can legitimately be near zero

    @pytest.mark.parametrize("M,N,K", [
        (1,   1,   1),
        (8,   8,   8),
        (16,  16,  16),
        (64,  64,  64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (33,  17,  5),    # non-power-of-2
        (1,   4096, 128), # skinny M (typical attention projection)
        (7,   7,   7),    # odd sizes
        (5,   15,  3),    # one partial 6x16 tile (direct path)
        (13,  33,  385),  # row/column tails past a 6x16 multiple, K > KC = 384 (packed)
        (97,  113, 769),  # K = 2*384 + 1: three KC panels on a 48 KiB-L1 core
        (431, 40,  50),   # M > MC = 426 on a 2.5 MiB-L2 core
    ])
    def test_sgemm_vs_numpy(self, M, N, K):
        A = rand_matrix(M, K)
        B = rand_matrix(K, N)
        C = np.zeros((M, N), dtype=np.float32)

        simd_ml_kernels.sgemm(A, B, C)
        C_ref = A.astype(np.float64) @ B.astype(np.float64)

        assert_close(C, C_ref, rtol=self.GEMM_TOL, atol=self.GEMM_ATOL,
                    label=f"GEMM [{M}×{N}×{K}]")

    @pytest.mark.parametrize("fmt", ["f16", "bf16"])
    @pytest.mark.parametrize("M,N,K", [
        (1, 1, 1), (7, 9, 8), (1, 4096, 128), (300, 16, 300), (50, 60, 70),   # direct path
        (64, 64, 64), (13, 33, 385), (97, 113, 769), (431, 40, 50), (256, 256, 256),
    ])
    def test_low_precision_vs_float64(self, fmt, M, N, K):
        """sgemm_f16 / sgemm_bf16: both conversions to FP32 are exact, so the
        reference is the float64 product of the converted inputs, and the
        bound is the FP32 GEMM's: |C - C64| <= 2(K+2)u * (|A||B|), u = 2^-24."""
        A, B = rand_matrix(M, K), rand_matrix(K, N)
        if fmt == "f16":
            Ah, Bh = A.astype(np.float16), B.astype(np.float16)
            Ar, Br = Ah.astype(np.float64), Bh.astype(np.float64)
            C = simd_ml_kernels.sgemm_f16(Ah, Bh)
        else:
            def to_bf16(x):   # round to nearest even
                u = x.view(np.uint32).astype(np.uint64)
                return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
            Ab, Bb = to_bf16(A), to_bf16(B)
            Ar = (Ab.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            Br = (Bb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            C = simd_ml_kernels.sgemm_bf16(Ab, Bb)
        bound = 2 * (K + 2) * 2.0 ** -24 * (np.abs(Ar) @ np.abs(Br))
        err = np.abs(C.astype(np.float64) - Ar @ Br)
        assert C.dtype == np.float32 and C.shape == (M, N)
        assert (err <= bound).all(), f"{fmt} [{M}x{N}x{K}]: max err/bound {np.max(err / bound):.3g}"

    @staticmethod
    def _to_fmt(x, fmt):
        """(array the kernel takes, its exact float64 values) for fp32 / f16 / bf16."""
        if fmt == "fp32":
            return x, x.astype(np.float64)
        if fmt == "f16":
            h = x.astype(np.float16)
            return h, h.astype(np.float64)
        u = x.view(np.uint32).astype(np.uint64)          # BF16, round to nearest even
        b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return b, (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    @staticmethod
    def _gemm_fn(fmt):
        return {"fp32": simd_ml_kernels.sgemm, "f16": simd_ml_kernels.sgemm_f16,
                "bf16": simd_ml_kernels.sgemm_bf16}[fmt]

    @pytest.mark.parametrize("fmt", ["fp32", "f16", "bf16"])
    @pytest.mark.parametrize("M,N,K", [
        (1, 37, 1), (1, 37, 15), (1, 37, 17), (1, 37, 33), (1, 1, 70), (1, 3, 70), (1, 5, 70),
        (1, 4096, 300), (1, 64, 4099), (1, 17, 70),       # decode GEMV: K, N tails
        (4, 33, 1027), (8, 45, 133),                      # GEMV, a few rows of x (M <= 8)
        (9, 45, 133), (97, 113, 769),                     # packed with transposed B
    ])
    def test_trans_b_vs_float64(self, fmt, M, N, K):
        """x @ W.T with W stored [N, K] (trans_b): the bound of the FP32 GEMM,
        |C - C64| <= 2(K+2)u * (|A||W|ᵀ), u = 2^-24, on the exact input values."""
        (A, Ar), (W, Wr) = self._to_fmt(rand_matrix(M, K), fmt), self._to_fmt(rand_matrix(N, K), fmt)
        C = self._gemm_fn(fmt)(A, W, trans_b=True)
        bound = 2 * (K + 2) * 2.0 ** -24 * (np.abs(Ar) @ np.abs(Wr).T)
        err = np.abs(C.astype(np.float64) - Ar @ Wr.T)
        assert C.dtype == np.float32 and C.shape == (M, N)
        assert (err <= bound).all(), f"{fmt} [{M}x{N}x{K}] tb: max err/bound {np.max(err / bound):.3g}"

    @pytest.mark.parametrize("fmt", ["fp32", "f16", "bf16"])
    def test_trans_b_unaligned_rows_alpha_beta(self, fmt):
        """GEMV on x and W that start one element past an aligned address
        (rows of W at odd 2- or 4-byte offsets), with alpha and beta."""
        M, N, K = 1, 21, 77
        (A, Ar), (W, Wr) = self._to_fmt(rand_matrix(M, K), fmt), self._to_fmt(rand_matrix(N, K), fmt)

        def shifted(x):
            buf = np.empty(x.size + 1, x.dtype)
            out = buf[1:].reshape(x.shape)
            out[...] = x
            return out
        A1, W1 = shifted(A), shifted(W)
        assert A1.ctypes.data % 8 != 0 and W1.ctypes.data % 8 != 0
        y0 = rand_matrix(M, N)
        y = y0.copy()
        self._gemm_fn(fmt)(A1, W1, C=y, alpha=-1.5, beta=0.5, trans_b=True)
        ref = -1.5 * (Ar @ Wr.T) + 0.5 * y0.astype(np.float64)
        bound = 2 * (K + 2) * 2.0 ** -24 * (1.5 * (np.abs(Ar) @ np.abs(Wr).T) + 0.5 * np.abs(y0))
        assert (np.abs(y - ref) <= bound).all()

    @pytest.mark.parametrize("fmt", ["fp32", "f16", "bf16"])
    def test_trans_b_nan_propagates(self, fmt):
        """A NaN in row j0 of W (inside the K tail) makes exactly y[:, j0] NaN."""
        M, N, K, j0 = 1, 9, 45, 6
        A, _ = self._to_fmt(np.full((M, K), 0.5, np.float32), fmt)
        Wf = np.full((N, K), 0.25, np.float32)
        Wf[j0, K - 2] = np.nan
        W, _ = self._to_fmt(Wf, fmt)
        y = self._gemm_fn(fmt)(A, W, trans_b=True)
        assert np.isnan(y[:, j0]).all()
        assert (np.delete(y, j0, axis=1) == 0.125 * K).all()

    @pytest.mark.parametrize("path", ["table", "fp32_convert"])
    @pytest.mark.parametrize("trans_b", [False, True])
    @pytest.mark.parametrize("M,N,K,zp", [
        (1, 1, 1, 0), (7, 9, 8, 128), (1, 4096, 128, 255), (5, 15, 3, 3), (13, 33, 385, 128),
        (97, 113, 1537, 77), (431, 40, 50, 128), (64, 64, 3077, 200),
    ])
    def test_u8s8_vs_float64(self, path, trans_b, M, N, K, zp):
        """sgemm_u8s8: exact reference in float64; documented bound
        |C - C64| <= 2(K+2)u * |a_scale*b_scales[j]| * (|A - zp| @ |B|)."""
        if path == "fp32_convert" and "avx2" not in simd_ml_kernels.build_info()["isa_usable"].split():
            pytest.skip("no AVX2 kernels on this CPU / build")
        rng = np.random.default_rng(M * 7 + N + K)
        A = rng.integers(0, 256, (M, K), dtype=np.uint8)
        B = rng.integers(-128, 128, (K, N), dtype=np.int8)
        scales = rng.uniform(-0.1, 0.1, N).astype(np.float32)
        a_scale = 0.0173
        fn = simd_ml_kernels.sgemm_u8s8 if path == "table" else simd_ml_kernels._core._sgemm_u8s8_convert
        C = fn(A, np.ascontiguousarray(B.T) if trans_b else B, a_scale, zp, scales, trans_b=trans_b)
        # Exact integer sums (int64 matmul, no BLAS; float64 BLAS was wrong under SDE -skx).
        ai, bi = A.astype(np.int64) - zp, B.astype(np.int64)
        sc = np.float64(np.float32(a_scale)) * scales.astype(np.float64)
        ref = (ai @ bi).astype(np.float64) * sc
        bound = 2 * (K + 2) * 2.0 ** -24 * (np.abs(ai) @ np.abs(bi)).astype(np.float64) * np.abs(sc) + 1e-30
        assert C.dtype == np.float32 and C.shape == (M, N)
        err = np.abs(C.astype(np.float64) - ref)
        assert (err <= bound).all(), f"[{M}x{N}x{K}] max err/bound {np.max(err / bound):.3g}"

    def test_u8s8_extreme_values_do_not_overflow(self):
        """A = 255, B = -128, zp = 0: the largest int32 block sums (K = 4096)."""
        M, N, K = 3, 33, 4096
        A = np.full((M, K), 255, np.uint8)
        B = np.full((K, N), -128, np.int8)
        C = simd_ml_kernels.sgemm_u8s8(A, B, 1.0, 0, np.ones(N, np.float32))
        np.testing.assert_array_equal(C, np.full((M, N), -255.0 * 128 * K, np.float32))

    @pytest.mark.parametrize("trans_a,trans_b", [(False, True), (True, False), (True, True)])
    @pytest.mark.parametrize("M,N,K", [
        (1, 300, 257),    # trans_b: dot-product path (M <= 8)
        (4, 33, 1027),    # dot path, K tail
        (5, 17, 9),       # trans_b packed, tiny
        (64, 64, 64),
        (97, 113, 769),   # three KC panels on a 48 KiB-L1 core
        (200, 190, 180),
    ])
    def test_sgemm_transposed(self, M, N, K, trans_a, trans_b):
        A = rand_matrix(K, M) if trans_a else rand_matrix(M, K)
        B = rand_matrix(N, K) if trans_b else rand_matrix(K, N)
        opA = (A.T if trans_a else A).astype(np.float64)
        opB = (B.T if trans_b else B).astype(np.float64)
        C = simd_ml_kernels.sgemm(A, B, trans_a=trans_a, trans_b=trans_b)
        assert C.shape == (M, N)
        assert_close(C, opA @ opB, rtol=self.GEMM_TOL, atol=self.GEMM_ATOL,
                     label=f"GEMM trans_a={trans_a} trans_b={trans_b} [{M}x{N}x{K}]")

    def test_sgemm_trans_b_linear_layer_alpha_beta(self):
        """y = 2 * x @ W.T + 0.5 * y with W stored [out, in], into a given C."""
        x, W = rand_matrix(8, 768), rand_matrix(3072, 768)
        y0 = rand_matrix(8, 3072)
        y = y0.copy()
        out = simd_ml_kernels.sgemm(x, W, y, alpha=2.0, beta=0.5, trans_b=True)
        assert out is y or np.shares_memory(out, y)
        ref = 2.0 * (x.astype(np.float64) @ W.T.astype(np.float64)) + 0.5 * y0
        assert_close(y, ref, rtol=self.GEMM_TOL, atol=self.GEMM_ATOL, label="x @ W.T")

    def test_sgemm_allocates_output_when_C_not_provided(self):
        A = rand_matrix(32, 64)
        B = rand_matrix(64, 32)
        C = simd_ml_kernels.sgemm(A, B)
        assert C.shape == (32, 32)
        assert C.dtype == np.float32
        C_ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
        assert np.allclose(C, C_ref, rtol=self.GEMM_TOL, atol=1e-4)

    def test_sgemm_alpha_beta(self):
        """Verify alpha scaling and beta accumulation."""
        M, N, K = 64, 64, 64
        A = rand_matrix(M, K)
        B = rand_matrix(K, N)
        C_init = rand_matrix(M, N)

        alpha, beta = 2.5, 0.3
        C_simd = C_init.copy()
        simd_ml_kernels.sgemm(A, B, C_simd, alpha=alpha, beta=beta)

        C_ref = alpha * (A.astype(np.float64) @ B.astype(np.float64)) \
              + beta * C_init.astype(np.float64)
        assert_close(C_simd, C_ref, rtol=self.GEMM_TOL, atol=self.GEMM_ATOL,
                    label="alpha/beta")

    def test_sgemm_beta_zero_overwrites_C(self):
        """beta=0 must completely overwrite C — not accumulate into garbage."""
        A = rand_matrix(32, 32)
        B = rand_matrix(32, 32)
        C = np.full((32, 32), 1e30, dtype=np.float32)   # poison value
        simd_ml_kernels.sgemm(A, B, C, beta=0.0)
        C_ref = (A @ B).astype(np.float32)
        assert np.allclose(C, C_ref, rtol=1e-3, atol=1e-4), \
            "beta=0 should overwrite C entirely"

    def test_sgemm_shape_mismatch_raises(self):
        A = rand_matrix(32, 64)
        B = rand_matrix(32, 32)  # wrong: K-dimension mismatch
        with pytest.raises(RuntimeError, match="Shape mismatch|ndim"):
            simd_ml_kernels.sgemm(A, B)

    def test_sgemm_non_contiguous_raises(self):
        A = rand_matrix(32, 32)
        B = rand_matrix(32, 32)
        with pytest.raises(RuntimeError, match="C-contiguous"):
            simd_ml_kernels.sgemm(A[::2], B)   # strided = not C-contiguous


class TestGEMMThreads:
    """sgemm with 1, 2 and 4 threads (2 and 4 only in OpenMP builds): within the
    float64 bound, and bitwise equal to the 1-thread result (each element of C
    is computed by one thread, in the same order)."""

    @pytest.fixture(autouse=True)
    def restore_threads(self):
        saved = simd_ml_kernels.get_num_threads()
        yield
        simd_ml_kernels.set_num_threads(saved)

    @pytest.mark.parametrize("threads", [1, 2, 4])
    @pytest.mark.parametrize("M,N,K,trans_a,trans_b", [
        (512, 512, 512, False, False),
        (13, 3000, 200, False, False),    # few rows: chunks cut along N too
        (1000, 37, 300, False, False),    # few columns, M crosses MC
        (97, 113, 769, False, False),     # three KC panels on a 48 KiB-L1 core
        (300, 5500, 23, False, False),    # N > NC = 2720 on an 8 MiB-L3 core: two column blocks
        (190, 210, 230, True, False),
        (190, 210, 230, False, True),
        (190, 210, 230, True, True),
        (64, 1024, 1024, False, True),    # x @ W.T, batch 64
    ])
    def test_sgemm_threads(self, M, N, K, trans_a, trans_b, threads):
        if threads > 1 and not simd_ml_kernels.build_info()["openmp"]:
            pytest.skip("built without OpenMP")
        A = rand_matrix(K, M) if trans_a else rand_matrix(M, K)
        B = rand_matrix(N, K) if trans_b else rand_matrix(K, N)
        simd_ml_kernels.set_num_threads(1)
        C1 = simd_ml_kernels.sgemm(A, B, trans_a=trans_a, trans_b=trans_b)
        simd_ml_kernels.set_num_threads(threads)
        C = simd_ml_kernels.sgemm(A, B, trans_a=trans_a, trans_b=trans_b)
        opA = (A.T if trans_a else A).astype(np.float64)
        opB = (B.T if trans_b else B).astype(np.float64)
        assert_close(C, opA @ opB, rtol=TestGEMM.GEMM_TOL, atol=TestGEMM.GEMM_ATOL,
                     label=f"GEMM {threads} threads [{M}x{N}x{K}] ta={trans_a} tb={trans_b}")
        assert np.array_equal(C.view(np.uint32), C1.view(np.uint32)), \
            f"{threads} threads differ from 1 thread"


class TestGEMVThreads:
    """Decode GEMV (x·Wᵀ in FP32/FP16/BF16, gemv_q8_0, gemv_q4_0) with 2, 3 and
    4 threads (OpenMP builds): bitwise equal to 1 thread, since each output row
    is computed by one thread with the same code. Shapes above the default
    bytes-per-thread minimum, and small ones with the minimum lowered to 1 byte
    so they split too (N with every remainder mod 8)."""

    @pytest.fixture(autouse=True)
    def restore_threads(self):
        saved = simd_ml_kernels.get_num_threads()
        min_bytes = simd_ml_kernels._core._gemv_min_bytes()
        yield
        simd_ml_kernels.set_num_threads(saved)
        simd_ml_kernels._core._gemv_min_bytes(min_bytes)

    @staticmethod
    def _all(x, W, n, k):
        x16 = x.astype(np.float16)
        W16 = W.astype(np.float16)
        xb = (x.view(np.uint32) >> 16).astype(np.uint16)
        Wb = (W.view(np.uint32) >> 16).astype(np.uint16)
        return [simd_ml_kernels.sgemm(x, W, trans_b=True),
                simd_ml_kernels.sgemm_f16(x16, W16, trans_b=True),
                simd_ml_kernels.sgemm_bf16(xb, Wb, trans_b=True),
                simd_ml_kernels.gemv_q8_0(x[0], simd_ml_kernels.quantize_q8_0(W), n, k),
                simd_ml_kernels.gemv_q4_0(x[0], simd_ml_kernels.quantize_q4_0(W), n, k)]

    @pytest.mark.parametrize("threads", [2, 3, 4])
    @pytest.mark.parametrize("N,K,force", [
        (1003, 2048, False),   # Q4_0 1.1 MB: split at the default minimum
        (4100, 1024, False),
        (9, 32, True), (31, 96, True), (257, 160, True), (2047, 64, True),
    ])
    def test_gemv_threads_bitwise(self, N, K, force, threads):
        if not simd_ml_kernels.build_info()["openmp"]:
            pytest.skip("built without OpenMP")
        if force:
            simd_ml_kernels._core._gemv_min_bytes(1.0)
        rng = np.random.default_rng(N * 7 + K)
        x = rng.standard_normal((1, K)).astype(np.float32)
        W = rng.standard_normal((N, K)).astype(np.float32)
        simd_ml_kernels.set_num_threads(1)
        ref = self._all(x, W, N, K)
        simd_ml_kernels.set_num_threads(threads)
        got = self._all(x, W, N, K)
        for name, a, b in zip(("fp32", "fp16", "bf16", "q8_0", "q4_0"), got, ref):
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), \
                f"{name} {N}x{K}: {threads} threads differ from 1 thread"


# ─── GeLU Tests ──────────────────────────────────────────────────────────────

# ─── Q8_0 (quantize_q8_0, gemv_q8_0) ──────────────────────────────────────────

def q8_0_quantize_numpy(w: np.ndarray) -> np.ndarray:
    """ggml's quantize_row_q8_0_ref in NumPy float32 arithmetic (IEEE, as the C
    reference): d = amax / 127, q = roundf(w * (1/d)) (half away from zero),
    scale stored as float16 (round to nearest even), little-endian."""
    k = w.shape[-1]
    b = w.astype(np.float32).reshape(-1, k // 32, 32)
    amax = np.abs(b).max(axis=2)
    d = (amax / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore", over="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    idv = np.where(np.isinf(idv), np.float32(0), idv)       # 1/d overflow: q = 0 (q8_0.hpp)
    v = (b * idv[:, :, None]).astype(np.float32).astype(np.float64)
    t = np.trunc(v)
    q = (t + (v - t >= 0.5) - (v - t <= -0.5)).astype(np.int8)
    out = np.empty(b.shape[:2] + (34,), np.uint8)
    out[:, :, :2] = d.astype("<f2").view(np.uint8).reshape(b.shape[:2] + (2,))
    out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(w.shape[:-1] + (k // 32 * 34,))


def _ggml_lib():
    """libggml-base.so of the llama.cpp build the bench uses (optional)."""
    path = os.environ.get("SIMD_ML_GGML_LIB")
    if not path and os.environ.get("SIMD_ML_GGML_TEST_BACKEND_OPS"):
        path = os.path.join(os.path.dirname(os.environ["SIMD_ML_GGML_TEST_BACKEND_OPS"]), "libggml-base.so")
    if not path or not os.path.exists(path):
        pytest.skip("set SIMD_ML_GGML_LIB (libggml-base.so) or SIMD_ML_GGML_TEST_BACKEND_OPS to compare with ggml")
    import ctypes

    return ctypes.CDLL(path)


class TestQ8_0:
    """quantize_q8_0 byte for byte against ggml's quantize_row_q8_0_ref
    (NumPy port, and the real ggml when available); gemv_q8_0 against the
    exact sum of its quantized operands (documented bound) and against the
    FP32 product (end-to-end bound from the quantization steps)."""

    @staticmethod
    def _special_rows() -> np.ndarray:
        rng = np.random.default_rng(11)
        w = rng.standard_normal((6, 128)).astype(np.float32)
        w[0, :32] = np.arange(-16, 16) + 0.5          # ties: amax 15.5, but exact halves...
        w[0, 31] = 127.0                                # ...with d = 1: k + 0.5 -> away from zero
        w[1, :32] = 0.0                                 # zero block: d = 0, q = 0
        w[1, 32:64] = rng.uniform(-1e-4, 1e-4, 32)      # d a float16 subnormal
        w[2, :] = rng.uniform(-8.3e6, 8.3e6, 128)       # large scales, below float16 overflow
        w[3, :32] = -w[3, :32]                          # negative values with a negative max
        w[4, :32] = np.float32(1.0)                     # constant block: q = 127
        w[5, :32] = np.where(np.arange(32) % 2, 1e-38, -1e-38).astype(np.float32)   # 1/d overflows: q = 0
        return w

    @pytest.mark.parametrize("shape", [(32,), (1, 32), (7, 64), (33, 4096), (3, 2, 96)])
    def test_quantize_matches_ggml_reference(self, shape):
        w = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32) * 3
        q = simd_ml_kernels.quantize_q8_0(w)
        assert q.dtype == np.uint8 and q.shape == shape[:-1] + (shape[-1] // 32 * 34,)
        np.testing.assert_array_equal(q, q8_0_quantize_numpy(w))

    def test_quantize_special_blocks(self):
        w = self._special_rows()
        q = simd_ml_kernels.quantize_q8_0(w)
        np.testing.assert_array_equal(q, q8_0_quantize_numpy(w))
        blk = q.reshape(6, 4, 34)
        assert blk[0, 0, 2:].view(np.int8).tolist()[:4] == [-16, -15, -14, -13]   # -15.5 -> -16
        assert (blk[1, 0] == 0).all()
        assert (blk[4, 0, 2:].view(np.int8) == 127).all()
        assert (blk[5, 0, 2:] == 0).all() and blk[5, 0, :2].view("<f2")[0] == 0

    def test_quantize_matches_real_ggml(self):
        import ctypes

        lib = _ggml_lib()
        w = np.concatenate([np.random.default_rng(5).standard_normal((40, 256)).astype(np.float32),
                            self._special_rows()[:5, :128].repeat(2, axis=1)])   # rows 0-4: no UB case
        q = simd_ml_kernels.quantize_q8_0(w)
        ref = np.empty_like(q)
        f = lib.quantize_row_q8_0_ref
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        for i in range(w.shape[0]):
            f(w[i].ctypes.data, ref[i].ctypes.data, w.shape[1])
        np.testing.assert_array_equal(q, ref)
        deq = np.empty(w.shape, np.float32)
        g = lib.dequantize_row_q8_0
        g.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        for i in range(w.shape[0]):
            g(q[i].ctypes.data, deq[i].ctypes.data, w.shape[1])
        d, qs = q8_0_blocks(q, w.shape[0], w.shape[1])
        np.testing.assert_array_equal(deq, (qs.astype(np.float32) * d.astype(np.float32)[:, :, None]).reshape(w.shape))

    @pytest.mark.parametrize("N,K", [(1, 32), (7, 64), (8, 96), (9, 128), (17, 352), (100, 4096), (33, 11008)])
    def test_gemv_vs_exact_quantized(self, N, K):
        rng = np.random.default_rng(N * 31 + K)
        w = rng.standard_normal((N, K)).astype(np.float32)
        x = (rng.standard_normal(K) * 4).astype(np.float32)
        wq = simd_ml_kernels.quantize_q8_0(w)
        y = simd_ml_kernels.gemv_q8_0(x, wq, N, K)
        assert y.dtype == np.float32 and y.shape == (N,)
        ref, mag = q8_0_reference(x, wq, N, K)
        bound = 2 * (K // 32 + 4) * 2.0 ** -24 * mag + 1e-300
        err = np.abs(y.astype(np.float64) - ref)
        assert (err <= bound).all(), f"[{N}x{K}] max err/bound {np.max(err / bound):.3g}"

    @pytest.mark.parametrize("N,K", [(64, 4096), (300, 2048)])
    def test_gemv_end_to_end_bound_vs_fp32(self, N, K):
        """|y - x·Wᵀ| <= sum_b [0.5622·dw_b·sum|x| + 0.5001·dx_b·sum|ŵ|] + the kernel bound:
        |w - ŵ| <= d·(0.5 + 2^-16 + 127·2^-11) per weight (rounding w/d, then d to
        float16, a normal float16 here), |x - x̂| <= dx·(0.5 + 2^-16)."""
        rng = np.random.default_rng(K + N)
        w = rng.standard_normal((N, K)).astype(np.float32)
        x = rng.standard_normal(K).astype(np.float32)
        wq = simd_ml_kernels.quantize_q8_0(w)
        y = simd_ml_kernels.gemv_q8_0(x, wq, N, K).astype(np.float64)
        dw, qw = q8_0_blocks(wq, N, K)
        assert (dw >= 2.0 ** -14).all()                  # normal float16 scales
        dx = q8_0_x_scales(x).astype(np.float64)
        what = np.abs(qw.astype(np.float64) * dw[:, :, None]).sum(axis=2)          # sum |ŵ| per block
        xs = np.abs(x.astype(np.float64)).reshape(-1, 32).sum(axis=1)               # sum |x| per block
        _, mag = q8_0_reference(x, wq, N, K)
        bound = (0.5622 * dw * xs[None, :] + 0.5001 * dx[None, :] * what).sum(axis=1) \
            + 2 * (K // 32 + 4) * 2.0 ** -24 * mag
        err = np.abs(y - w.astype(np.float64) @ x.astype(np.float64))
        assert (err <= bound).all(), f"max err/bound {np.max(err / bound):.3g}"

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_gemv_nonfinite_x_gives_nan(self, bad):
        w = np.full((9, 64), 0.25, np.float32)
        x = np.ones(64, np.float32)
        x[37] = bad
        y = simd_ml_kernels.gemv_q8_0(x, simd_ml_kernels.quantize_q8_0(w), 9, 64)
        assert np.isnan(y).all()

    def test_gemv_k0_and_n0(self):
        assert (simd_ml_kernels.gemv_q8_0(np.ones(0, np.float32), np.zeros(0, np.uint8), 5, 0) == 0).all()
        assert simd_ml_kernels.gemv_q8_0(np.ones(32, np.float32), np.zeros(0, np.uint8), 0, 32).shape == (0,)

    def test_gemv_unaligned_wq(self):
        """Wq at an odd address (a slice of a larger buffer)."""
        N, K = 13, 160
        rng = np.random.default_rng(3)
        w = rng.standard_normal((N, K)).astype(np.float32)
        x = rng.standard_normal(K).astype(np.float32)
        wq = simd_ml_kernels.quantize_q8_0(w).ravel()
        buf = np.empty(wq.size + 1, np.uint8)
        buf[1:] = wq
        np.testing.assert_array_equal(simd_ml_kernels.gemv_q8_0(x, buf[1:], N, K),
                                      simd_ml_kernels.gemv_q8_0(x, wq, N, K))


# ─── Q4_0 (quantize_q4_0, gemv_q4_0) ──────────────────────────────────────────

Q4_0_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "q4_0_ggml_v0.7.0.npz")


def q4_0_quantize_numpy(w: np.ndarray) -> np.ndarray:
    """ggml's quantize_row_q4_0_ref in NumPy float32 arithmetic (each step
    rounded, as the C reference built for baseline x86-64): max = the first
    value of largest magnitude, d = max / -8, q = min(15, trunc(w * (1/d) + 8.5)),
    scale stored as float16 (round to nearest even), little-endian; byte j =
    q[j] | q[16 + j] << 4."""
    k = w.shape[-1]
    b = w.astype(np.float32).reshape(-1, k // 32, 32)
    a = np.abs(b)
    amax = a.max(axis=2)
    first = np.take_along_axis(b, a.argmax(axis=2)[:, :, None], axis=2)[:, :, 0]
    mx = np.where(amax > 0, first, np.float32(0)).astype(np.float32)   # all-zero block: max stays +0.0
    d = (mx / np.float32(-8)).astype(np.float32)
    with np.errstate(divide="ignore", over="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    idv = np.where(np.isinf(idv), np.float32(0), idv)       # 1/d overflow: q = 8 (q4_0.hpp)
    v = (b * idv[:, :, None]).astype(np.float32)
    t = (v + np.float32(8.5)).astype(np.float32)
    q = np.minimum(np.trunc(t), 15).astype(np.uint8)
    out = np.empty(b.shape[:2] + (18,), np.uint8)
    out[:, :, :2] = d.astype("<f2").view(np.uint8).reshape(b.shape[:2] + (2,))
    out[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)
    return out.reshape(w.shape[:-1] + (k // 32 * 18,))


def q4_0_fixture_input() -> np.ndarray:
    """The input of the committed llama.cpp fixture (and of its generator):
    20 rows of 256 float32 values, random rows at several scales and rows
    that exercise every rule of quantize_row_q4_0_ref."""
    rng = np.random.default_rng(2026)
    w = rng.standard_normal((20, 256)).astype(np.float32)
    w[:12] *= np.array([1, 3, 0.01, 100, 1e-3, 0.5, 7, 1, 1, 2, 40, 1e4], np.float32)[:, None]
    w[12] = np.tile(np.arange(-8, 8, 0.5, dtype=np.float32), 8)          # max -8: w + 8.5 on k + 0.5 and k
    w[12, 32:64] = -w[12, 32:64]                                          # max +8: d = -1
    w[12, 64:] *= np.repeat(np.float32([0.25, 3, 1e-2, 7, 11, 0.125]), 32)
    w[13, :] = rng.uniform(-1, 1, 256).astype(np.float32)
    w[13, 5], w[13, 9] = 2.0, -2.0                                         # equal magnitudes: the first sets the sign
    w[13, 35], w[13, 62] = -2.0, 2.0
    w[13, 64 + 20], w[13, 64 + 3] = 2.0, -2.0                              # first by position, not by order written
    w[14, :32] = 0.0                                                       # zero block: d = -0.0, q = 8
    w[14, 32:64] = -0.0
    w[14, 64:96] = np.where(np.arange(32) % 3 == 0, 0.0, -0.0)
    w[15, :128] = rng.uniform(-1e-5, 1e-5, 128).astype(np.float32)        # d a float16 subnormal
    w[15, 128:] = rng.uniform(-1e-36, 1e-36, 128).astype(np.float32)      # d far below float16 (scale 0), 1/d finite
    w[16, :] = rng.uniform(-5.2e5, 5.2e5, 256).astype(np.float32)         # large scales, below float16 overflow
    w[17, :] = np.round(rng.uniform(-8, 8, 256)).astype(np.float32) * 0.5 # many exact halves
    w[18, :] = -np.abs(w[18, :])                                          # all negative
    w[19, ::2] = np.float32(7.0)                                          # +7 next to a -8 max: q = 15
    w[19, 0::32] = np.float32(-8.0)
    return w


def write_q4_0_fixture(lib_path: str, out_path: str = Q4_0_FIXTURE) -> None:
    """Write the Q4_0 fixture: llama.cpp's own quantization (ggml_quantize_chunk,
    the call llama-quantize makes; Q4_0 without an importance matrix is
    quantize_row_q4_0_ref) of q4_0_fixture_input(), run in libggml-base.so."""
    import ctypes
    import subprocess

    lib = ctypes.CDLL(lib_path)
    f = lib.ggml_quantize_chunk
    f.restype = ctypes.c_size_t
    f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                  ctypes.c_int64, ctypes.c_void_p]
    w = q4_0_fixture_input()
    out = np.zeros((w.shape[0], w.shape[1] // 32 * 18), np.uint8)
    n = f(2, w.ctypes.data, out.ctypes.data, 0, w.shape[0], w.shape[1], None)   # 2 = GGML_TYPE_Q4_0
    assert n == out.size, n
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(lib_path))))
    git = lambda *a: subprocess.run(["git", "-C", src, *a], capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, w=w, blocks=out,
                        source=np.array(f"llama.cpp {git('describe', '--tags')} {git('rev-parse', 'HEAD')}: "
                                        "ggml_quantize_chunk(GGML_TYPE_Q4_0, w, out, 0, 20, 256, NULL)"))


class TestQ4_0:
    """quantize_q4_0 byte for byte against ggml's quantize_row_q4_0_ref
    (NumPy port; blocks written by llama.cpp v0.7.0, committed; the live
    ggml library when available); gemv_q4_0 against the exact sum of its
    quantized operands (documented bound) and against the FP32 product
    (end-to-end bound from the quantization steps)."""

    @pytest.mark.parametrize("shape", [(32,), (1, 32), (7, 64), (33, 4096), (3, 2, 96)])
    def test_quantize_matches_ggml_reference(self, shape):
        w = np.random.default_rng(sum(shape) + 1).standard_normal(shape).astype(np.float32) * 3
        q = simd_ml_kernels.quantize_q4_0(w)
        assert q.dtype == np.uint8 and q.shape == shape[:-1] + (shape[-1] // 32 * 18,)
        np.testing.assert_array_equal(q, q4_0_quantize_numpy(w))

    def test_quantize_matches_llama_cpp_fixture(self):
        """Blocks written by llama.cpp's own quantization (v0.7.0, 620d2225),
        generated once on the fixture's input with:
            python tests/test_precision.py write-q4_0-fixture \\
                ~/code/llama.cpp/build/bin/libggml-base.so
        (ggml_quantize_chunk(GGML_TYPE_Q4_0, ...), as llama-quantize calls it)."""
        fx = np.load(Q4_0_FIXTURE)
        np.testing.assert_array_equal(fx["w"], q4_0_fixture_input())   # the input is what the generator used
        q = simd_ml_kernels.quantize_q4_0(fx["w"])
        np.testing.assert_array_equal(q, fx["blocks"])
        np.testing.assert_array_equal(q4_0_quantize_numpy(fx["w"]), fx["blocks"])
        blk = fx["blocks"].reshape(20, 8, 18)
        assert (blk[14, 0, :2].view("<f2") == 0).all() and blk[14, 0, :2].view("<u2")[0] == 0x8000   # -0.0
        assert (blk[14, 0, 2:] == 0x88).all()
        assert (blk[19, 0, 2:] & 0x0F == 0).sum() >= 1 and (blk[19, 0, 2:] >> 4 == 15).any()

    def test_quantize_special_blocks(self):
        w = np.zeros((2, 64), np.float32)
        w[0, :32] = np.where(np.arange(32) % 2, 1e-44, -1e-44).astype(np.float32)   # 1/d overflows: q = 8
        w[0, 32:] = 1.0                                                              # constant: max +1, d = -1/8
        w[1, :] = np.random.default_rng(1).standard_normal(64).astype(np.float32)
        q = simd_ml_kernels.quantize_q4_0(w)
        np.testing.assert_array_equal(q, q4_0_quantize_numpy(w))
        blk = q.reshape(2, 2, 18)
        assert (blk[0, 0, 2:] == 0x88).all() and blk[0, 0, :2].view("<f2")[0] == 0
        assert (blk[0, 1, 2:] == 0x00).all() and blk[0, 1, :2].view("<f2")[0] == np.float16(-0.125)

    def test_quantize_matches_real_ggml(self):
        import ctypes

        lib = _ggml_lib()
        w = np.concatenate([np.random.default_rng(6).standard_normal((40, 256)).astype(np.float32),
                            q4_0_fixture_input()])
        q = simd_ml_kernels.quantize_q4_0(w)
        ref = np.empty_like(q)
        f = lib.quantize_row_q4_0_ref
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        for i in range(w.shape[0]):
            f(w[i].ctypes.data, ref[i].ctypes.data, w.shape[1])
        np.testing.assert_array_equal(q, ref)
        deq = np.empty(w.shape, np.float32)
        g = lib.dequantize_row_q4_0
        g.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        for i in range(w.shape[0]):
            g(q[i].ctypes.data, deq[i].ctypes.data, w.shape[1])
        d, qs = q4_0_blocks(q, w.shape[0], w.shape[1])
        np.testing.assert_array_equal(deq, (qs.astype(np.float32) * d.astype(np.float32)[:, :, None]).reshape(w.shape))

    @pytest.mark.parametrize("N,K", [(1, 32), (7, 64), (8, 96), (9, 128), (17, 352), (100, 4096), (33, 11008)])
    def test_gemv_vs_exact_quantized(self, N, K):
        rng = np.random.default_rng(N * 37 + K)
        w = rng.standard_normal((N, K)).astype(np.float32)
        x = (rng.standard_normal(K) * 4).astype(np.float32)
        wq = simd_ml_kernels.quantize_q4_0(w)
        y = simd_ml_kernels.gemv_q4_0(x, wq, N, K)
        assert y.dtype == np.float32 and y.shape == (N,)
        ref, mag = q4_0_reference(x, wq, N, K)
        bound = 2 * (K // 32 + 4) * 2.0 ** -24 * mag + 1e-300
        err = np.abs(y.astype(np.float64) - ref)
        assert (err <= bound).all(), f"[{N}x{K}] max err/bound {np.max(err / bound):.3g}"

    @pytest.mark.parametrize("N,K", [(64, 4096), (300, 2048)])
    def test_gemv_end_to_end_bound_vs_fp32(self, N, K):
        """|y - x·Wᵀ| <= sum_b [1.0045·dw_b·sum|x| + 0.5001·dx_b·sum|ŵ|] + the kernel bound:
        |w - ŵ| <= d·(1 + 8·2^-11) per weight (rounding w/d to q - 8, clamped at
        +7 for values >= 7.5·d, then d to float16, a normal float16 here),
        |x - x̂| <= dx·(0.5 + 2^-16)."""
        rng = np.random.default_rng(K + N + 1)
        w = rng.standard_normal((N, K)).astype(np.float32)
        x = rng.standard_normal(K).astype(np.float32)
        wq = simd_ml_kernels.quantize_q4_0(w)
        y = simd_ml_kernels.gemv_q4_0(x, wq, N, K).astype(np.float64)
        dw, qw = q4_0_blocks(wq, N, K)
        assert (np.abs(dw) >= 2.0 ** -14).all()          # normal float16 scales
        dx = q8_0_x_scales(x).astype(np.float64)
        what = np.abs(qw.astype(np.float64) * dw[:, :, None]).sum(axis=2)          # sum |ŵ| per block
        xs = np.abs(x.astype(np.float64)).reshape(-1, 32).sum(axis=1)               # sum |x| per block
        _, mag = q4_0_reference(x, wq, N, K)
        bound = (1.0045 * np.abs(dw) * xs[None, :] + 0.5001 * dx[None, :] * what).sum(axis=1) \
            + 2 * (K // 32 + 4) * 2.0 ** -24 * mag
        err = np.abs(y - w.astype(np.float64) @ x.astype(np.float64))
        assert (err <= bound).all(), f"max err/bound {np.max(err / bound):.3g}"

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_gemv_nonfinite_x_gives_nan(self, bad):
        w = np.full((9, 64), 0.25, np.float32)
        x = np.ones(64, np.float32)
        x[37] = bad
        y = simd_ml_kernels.gemv_q4_0(x, simd_ml_kernels.quantize_q4_0(w), 9, 64)
        assert np.isnan(y).all()

    def test_gemv_k0_and_n0(self):
        assert (simd_ml_kernels.gemv_q4_0(np.ones(0, np.float32), np.zeros(0, np.uint8), 5, 0) == 0).all()
        assert simd_ml_kernels.gemv_q4_0(np.ones(32, np.float32), np.zeros(0, np.uint8), 0, 32).shape == (0,)

    def test_gemv_unaligned(self):
        """Wq at an odd address and x at a 4-byte (not 32-byte) offset."""
        N, K = 13, 160
        rng = np.random.default_rng(4)
        w = rng.standard_normal((N, K)).astype(np.float32)
        x = rng.standard_normal(K).astype(np.float32)
        wq = simd_ml_kernels.quantize_q4_0(w).ravel()
        buf = np.empty(wq.size + 1, np.uint8)
        buf[1:] = wq
        xbuf = np.empty(K + 1, np.float32)
        xbuf[1:] = x
        want = simd_ml_kernels.gemv_q4_0(x, wq, N, K)
        np.testing.assert_array_equal(simd_ml_kernels.gemv_q4_0(xbuf[1:], buf[1:], N, K), want)
        ref, mag = q4_0_reference(x, wq, N, K)
        assert (np.abs(want - ref) <= 2 * (K // 32 + 4) * 2.0 ** -24 * mag + 1e-300).all()


class TestGeLU:
    """GeLU, both variants, against float64 evaluations of the same formula.

    The bounds asserted here are the ones quoted in the gelu() docstring.
    """

    GELU_TOL = 1e-4    # relative component for random-input comparisons
    GELU_ATOL = 1e-6   # absolute floor: GeLU crosses zero at x=0, needs an absolute bound
                       # near the crossing (see assert_close docstring for why this matters)
    SWEEP = np.linspace(-10.0, 10.0, 400_001).astype(np.float32)

    @staticmethod
    def _erf_ref(x):
        x64 = np.asarray(x, dtype=np.float64)
        erf = np.vectorize(math.erf)(x64 / math.sqrt(2.0))
        return 0.5 * x64 * (1.0 + erf)

    def test_default_is_exact(self):
        x = RNG.standard_normal(1000).astype(np.float32)
        np.testing.assert_array_equal(simd_ml_kernels.gelu(x),
                                      simd_ml_kernels.gelu(x, approximate="none"))

    def test_exact_sweep_bound(self):
        x = self.SWEEP
        err = np.abs(simd_ml_kernels.gelu(x).astype(np.float64) - self._erf_ref(x))
        assert err.max() <= 2e-6, f"max abs err {err.max():.3e} on [-10, 10]"
        core = np.abs(x) <= 3.0
        assert err[core].max() <= 1e-6, f"max abs err {err[core].max():.3e} on [-3, 3]"

    def test_tanh_sweep_bound(self):
        x = self.SWEEP
        out = simd_ml_kernels.gelu(x, approximate="tanh").astype(np.float64)
        err = np.abs(out - gelu_tanh_numpy(x))
        assert err.max() <= 2e-6, f"max abs err {err.max():.3e} on [-10, 10]"

    def test_tanh_vs_exact_formula_gap(self):
        """The two formulas differ by up to ~4.7e-4; documents why both exist."""
        x = self.SWEEP
        gap = np.abs(simd_ml_kernels.gelu(x, approximate="tanh").astype(np.float64)
                     - self._erf_ref(x))
        assert 4e-4 < gap.max() < 5e-4

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 16, 64, 1024, 65536])
    def test_random_inputs_all_sizes(self, approximate, n):
        x = RNG.uniform(-3.0, 3.0, size=n).astype(np.float32)
        ref = self._erf_ref(x) if approximate == "none" else gelu_tanh_numpy(x)
        out = simd_ml_kernels.gelu(x, approximate=approximate)
        assert_close(out, ref, rtol=self.GELU_TOL, atol=self.GELU_ATOL,
                     label=f"GeLU({approximate}) n={n}")

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_tail_matches_vector_body_bitwise(self, approximate):
        """Elements in the last n % 8 go through the same kernel (masked), so
        a value's result does not depend on its position in the array."""
        x = RNG.standard_normal(16).astype(np.float32)
        full = simd_ml_kernels.gelu(x, approximate=approximate)       # all in 8-wide body
        part = simd_ml_kernels.gelu(x[:13].copy(), approximate=approximate)  # 8..12 in tail
        np.testing.assert_array_equal(part, full[:13])

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    @pytest.mark.parametrize("offset", [0, 9])   # vector body / masked tail
    def test_special_values(self, approximate, offset):
        special = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0, 1e-40, -1e-40],
                           dtype=np.float32)
        x = np.zeros(offset + special.size, np.float32)
        x[offset:] = special
        out = simd_ml_kernels.gelu(x, approximate=approximate)[offset:]
        assert out[0] == np.inf
        assert out[1] == 0.0
        assert np.isnan(out[2])
        assert out[3] == 0.0 and out[4] == 0.0
        assert np.all(np.abs(out[5:]) <= 1e-40)

    def test_invalid_approximate(self):
        with pytest.raises(simd_ml_kernels.ArgumentError, match="approximate"):
            simd_ml_kernels.gelu(np.ones(4, np.float32), approximate="erf")

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_inplace_equals_outofplace(self, approximate):
        x = RNG.standard_normal(8195).astype(np.float32)
        out_oop = simd_ml_kernels.gelu(x, approximate=approximate)
        x_ip = x.copy()
        simd_ml_kernels.gelu_inplace(x_ip, approximate=approximate)
        np.testing.assert_array_equal(out_oop, x_ip)

    def test_output_shape_preserved(self):
        for shape in [(8,), (4, 16), (2, 3, 32)]:
            x = rand_array(int(np.prod(shape))).reshape(shape)
            assert simd_ml_kernels.gelu(x).shape == shape

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_vs_pytorch(self, approximate):
        x_np = RNG.standard_normal(4096).astype(np.float32)
        ref = F.gelu(torch.from_numpy(x_np), approximate=approximate).numpy()
        got = simd_ml_kernels.gelu(x_np, approximate=approximate)
        assert_close(got, ref, rtol=self.GELU_TOL, atol=self.GELU_ATOL,
                     label=f"vs PyTorch GeLU({approximate})")


# ─── ReLU Tests ──────────────────────────────────────────────────────────────

class TestReLU:
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 64, 1024, 65537])
    def test_relu_vs_numpy(self, n):
        x = RNG.uniform(-3.0, 3.0, size=n).astype(np.float32)
        ref = relu_numpy(x).astype(np.float32)
        out = simd_ml_kernels.relu(x)
        # ReLU is exact (just a max with zero) — should match exactly
        assert np.array_equal(out, ref), f"ReLU n={n}: mismatch"

    def test_relu_all_negative(self):
        x = np.full(64, -5.0, dtype=np.float32)
        assert np.all(simd_ml_kernels.relu(x) == 0.0)

    def test_relu_all_positive(self):
        x = np.arange(1, 65, dtype=np.float32)
        out = simd_ml_kernels.relu(x)
        assert np.array_equal(out, x)


# ─── SiLU Tests ──────────────────────────────────────────────────────────────

class TestSiLU:
    # Documented bound (silu_avx2.cpp, docstring): 1e-35 + 1e-6*|ref| over every
    # finite float; measured 2.9e-7 relative on the AVX2 table.
    SILU_TOL = 1e-6
    SILU_ATOL = 1e-35

    @pytest.mark.parametrize("n", [1, 7, 8, 9, 64, 1024, 65536])
    @pytest.mark.parametrize("bound", [4.0, 200.0])  # 200: past the x < -88 cut-off
    def test_silu_vs_numpy(self, n, bound):
        x = RNG.uniform(-bound, bound, size=n).astype(np.float32)
        ref = silu_numpy(x)
        out = simd_ml_kernels.silu(x)
        assert_close(out, ref, rtol=self.SILU_TOL, atol=self.SILU_ATOL,
                    label=f"SiLU n={n} on [-{bound}, {bound}]")

    def test_silu_every_binade(self):
        """±m·2^e for e in [-126, 127]. The AVX2 kernel used to clamp tanh(x/2),
        so its error grew without bound below x = -16 (5e31 at -FLT_MAX)."""
        e = np.arange(-126, 128)
        mags = np.concatenate([np.ldexp(m, e) for m in (1.0, 1.3, 1.9999999)])
        x = np.concatenate([mags, -mags, [-88.5, -88.0, -87.5, -103.0, 88.5]]).astype(np.float32)
        with np.errstate(over="ignore"):
            ref = silu_numpy(x)
        assert_close(simd_ml_kernels.silu(x), ref, rtol=self.SILU_TOL,
                     atol=self.SILU_ATOL, label="SiLU every binade")

    @pytest.mark.parametrize("offset", [0, 9])   # vector body / masked tail
    def test_silu_special_values(self, offset):
        special = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float32)
        x = np.ones(offset + special.size, np.float32)
        x[offset:] = special
        out = simd_ml_kernels.silu(x)[offset:]
        assert out[0] == np.inf
        assert np.isnan(out[1])  # -inf / (1 + inf), as PyTorch and the scalar kernel
        assert np.isnan(out[2])
        assert out[3] == 0.0 and not np.signbit(out[3]) and out[4] == 0.0

    def test_silu_tail_matches_vector_body_bitwise(self):
        x = RNG.uniform(-30.0, 30.0, 16).astype(np.float32)
        np.testing.assert_array_equal(simd_ml_kernels.silu(x[:13].copy()),
                                      simd_ml_kernels.silu(x)[:13])

    def test_silu_zero_input(self):
        x = np.zeros(64, dtype=np.float32)
        out = simd_ml_kernels.silu(x)
        assert np.allclose(out, 0.0, atol=1e-7)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_silu_vs_pytorch(self):
        x_np = RNG.standard_normal(4096).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        ref = F.silu(x_pt).numpy()
        got = simd_ml_kernels.silu(x_np)
        assert_close(got, ref, rtol=self.SILU_TOL, atol=self.SILU_ATOL,
                    label="vs PyTorch SiLU")


# ─── Softmax Tests ────────────────────────────────────────────────────────────

class TestSoftmax:
    @pytest.mark.parametrize("shape", [(8,), (64,), (512,), (4, 128), (16, 64)])
    def test_softmax_sums_to_one(self, shape):
        x = RNG.standard_normal(shape).astype(np.float32)
        out = simd_ml_kernels.softmax(x)
        row_sums = out.reshape(-1, shape[-1]).sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"Softmax row sums not 1.0 for shape {shape}: {row_sums}"

    @pytest.mark.parametrize("n", [4, 16, 64, 512])
    def test_softmax_vs_numpy(self, n):
        x = RNG.standard_normal(n).astype(np.float32)
        ref = softmax_numpy(x)
        out = simd_ml_kernels.softmax(x)
        max_err = float(np.max(np.abs(out.astype(np.float64) - ref)))
        assert max_err < 1e-5, f"Softmax n={n}: max_abs_err={max_err:.2e}"

    def test_softmax_argmax_preserved(self):
        """argmax(softmax(x)) == argmax(x)."""
        x = RNG.standard_normal(256).astype(np.float32)
        out = simd_ml_kernels.softmax(x)
        assert np.argmax(out) == np.argmax(x)

    def test_softmax_numerical_stability_large_values(self):
        """Input with very large values should not produce NaN/Inf."""
        x = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)
        out = simd_ml_kernels.softmax(x)
        assert np.all(np.isfinite(out)), "Softmax must be numerically stable for large inputs"
        assert np.allclose(out.sum(), 1.0, atol=1e-6)

    # Rows whose result is NaN in torch.softmax (max subtraction): a NaN
    # anywhere, any +inf (inf - inf), or every element -inf. n = 4 runs only the
    # scalar tail, 8 / 16 only the vector body, 9 / 19 both.
    @pytest.mark.parametrize("n", [1, 4, 8, 9, 16, 19])
    @pytest.mark.parametrize("kind", ["all_neg_inf", "one_nan", "one_pos_inf"])
    def test_softmax_nan_rows(self, n, kind):
        x = RNG.standard_normal(n).astype(np.float32)
        if kind == "all_neg_inf":
            x[:] = -np.inf
        else:
            x[n // 2] = np.nan if kind == "one_nan" else np.inf
        assert np.isnan(simd_ml_kernels.softmax(x)).all()

    @pytest.mark.parametrize("n", [4, 9, 19])
    def test_softmax_partial_neg_inf_mask(self, n):
        x = RNG.standard_normal(n).astype(np.float32)
        x[::2] = -np.inf
        out = simd_ml_kernels.softmax(x)
        np.testing.assert_allclose(out, softmax_numpy(x), atol=1e-6)
        assert (out[::2] == 0.0).all()

    def test_softmax_rows_are_independent(self):
        x = RNG.standard_normal((3, 9)).astype(np.float32)
        x[1, :] = -np.inf
        out = simd_ml_kernels.softmax(x)
        assert np.isnan(out[1]).all()
        np.testing.assert_allclose(out[[0, 2]], softmax_numpy(x[[0, 2]]), atol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_softmax_special_rows_match_pytorch(self):
        rows = np.array([[-np.inf] * 9,
                         [0.0, np.nan] + [1.0] * 7,
                         [np.inf, 0.0] + [1.0] * 7,
                         [-np.inf, 0.0] + [1.0] * 7], dtype=np.float32)
        ref = F.softmax(torch.from_numpy(rows), dim=-1).numpy()
        np.testing.assert_allclose(simd_ml_kernels.softmax(rows), ref, atol=1e-6, equal_nan=True)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_softmax_vs_pytorch_2d(self):
        x = RNG.standard_normal((8, 64)).astype(np.float32)
        ref = F.softmax(torch.from_numpy(x), dim=-1).numpy()
        out = simd_ml_kernels.softmax(x, axis=-1)
        assert np.allclose(out, ref, atol=1e-5)


# ─── LayerNorm Tests ──────────────────────────────────────────────────────────

class TestLayerNorm:
    LN_TOL = 1e-4
    LN_ATOL = 1e-5  # normalized output crosses zero regularly — see assert_close docstring

    @pytest.mark.parametrize("use_gamma", [False, True])
    @pytest.mark.parametrize("use_beta", [False, True])
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 17, 64, 768, 1027])
    def test_layer_norm_affine_combinations(self, n, use_gamma, use_beta):
        """Every gamma/beta combination matches the float64 reference.

        n covers the scalar tail alone, whole 8-wide AVX2 blocks, and both.
        Inputs have mean 50 and std 100; beta has unit scale, so dropping it
        would be off by O(1), far outside atol=1e-5 / rtol=1e-4.
        """
        x     = (50.0 + 100.0 * RNG.standard_normal((3, n))).astype(np.float32)
        gamma = RNG.standard_normal(n).astype(np.float32) if use_gamma else None
        beta  = RNG.standard_normal(n).astype(np.float32) if use_beta else None

        ref = layer_norm_numpy(x, gamma, beta)
        out = simd_ml_kernels.layer_norm(x, gamma=gamma, beta=beta)

        assert_close(out, ref, rtol=self.LN_TOL, atol=self.LN_ATOL,
                    label=f"LayerNorm n={n} gamma={use_gamma} beta={use_beta}")

    @pytest.mark.parametrize("n", [1027, 65536])
    @pytest.mark.parametrize("mean,std,outlier", [
        (30.0, 1.0, False), (-3e4, 1e3, False), (0.3, 1e-2, False), (0.0, 1.0, True)])
    def test_layer_norm_conditioning(self, n, mean, std, outlier):
        """The documented bound holds for |mean| <= 30*std, at row lengths where
        float partial sums would drift, and with x[0] far from the mean (the
        statistics pass sums x - x[0])."""
        x = (mean + std * RNG.standard_normal((2, n))).astype(np.float32)
        if outlier:
            x[:, 0] = 1e3
        gamma = RNG.uniform(0.0, 2.0, n).astype(np.float32)
        beta = RNG.uniform(-1.0, 1.0, n).astype(np.float32)
        assert_close(simd_ml_kernels.layer_norm(x, gamma=gamma, beta=beta),
                     layer_norm_numpy(x, gamma, beta), rtol=self.LN_TOL, atol=self.LN_ATOL,
                     label=f"LayerNorm n={n} mean={mean} std={std} outlier={outlier}")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_layer_norm_vs_pytorch(self):
        n     = 256
        x     = RNG.standard_normal(n).astype(np.float32)
        gamma = np.ones(n, dtype=np.float32)
        beta  = np.zeros(n, dtype=np.float32)
        ref   = torch.nn.functional.layer_norm(
                    torch.from_numpy(x), [n],
                    weight=torch.from_numpy(gamma),
                    bias=torch.from_numpy(beta)).numpy()
        out   = simd_ml_kernels.layer_norm(x, gamma=gamma, beta=beta)
        assert_close(out, ref, rtol=self.LN_TOL, atol=self.LN_ATOL,
                    label="vs PyTorch LayerNorm")


# ─── RMSNorm and SiLU·up Tests ────────────────────────────────────────────────

def rms_norm_numpy(x, weight=None, eps=1e-5):
    x64 = x.astype(np.float64)
    r = 1.0 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + eps)
    return x64 * r * (1.0 if weight is None else weight.astype(np.float64))


class TestRMSNorm:
    # Documented bound (rms_norm_avx2.cpp, docstring): 1e-30 + 1e-6*|ref|.
    RTOL = 1e-6
    ATOL = 1e-30

    @pytest.mark.parametrize("use_weight", [False, True])
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 31, 32, 33, 1023, 1024, 1025, 2048, 4099])
    def test_rms_norm_vs_float64(self, n, use_weight):
        x = (3.0 * RNG.standard_normal((3, n))).astype(np.float32)
        w = RNG.uniform(0.5, 1.5, n).astype(np.float32) if use_weight else None
        assert_close(simd_ml_kernels.rms_norm(x, w), rms_norm_numpy(x, w), rtol=self.RTOL,
                     atol=self.ATOL, label=f"RMSNorm n={n} weight={use_weight}")

    @pytest.mark.parametrize("scale", [1e-20, 1e-3, 1e15])
    def test_rms_norm_scales(self, scale):
        """Rows far below sqrt(eps) (eps dominates) and far above 1."""
        x = (scale * RNG.standard_normal(2048)).astype(np.float32)
        w = RNG.uniform(0.5, 1.5, 2048).astype(np.float32)
        assert_close(simd_ml_kernels.rms_norm(x, w, eps=1e-5), rms_norm_numpy(x, w, 1e-5),
                     rtol=self.RTOL, atol=self.ATOL, label=f"RMSNorm scale={scale}")

    def test_rms_norm_unaligned(self):
        buf = (2.0 * RNG.standard_normal(2049)).astype(np.float32)
        x = buf[1:]   # offset 4 bytes from the allocation
        w = RNG.uniform(0.5, 1.5, 2048).astype(np.float32)
        assert_close(simd_ml_kernels.rms_norm(x, w), rms_norm_numpy(x, w), rtol=self.RTOL,
                     atol=self.ATOL, label="RMSNorm unaligned")

    def test_rms_norm_special_values(self):
        x = RNG.standard_normal((3, 37)).astype(np.float32)
        x[0, 20] = np.nan
        x[1, 5] = np.inf
        x[2] = 0.0
        out = simd_ml_kernels.rms_norm(x)
        assert np.isnan(out[0]).all()
        assert np.isnan(out[1, 5]) and (np.delete(out[1], 5) == 0.0).all()
        assert (out[2] == 0.0).all()

    def test_rms_norm_matches_llama_numpy_order(self):
        """The Llama example's NumPy RMSNorm (transformers' order) agrees to
        the bound; the kernel only sums x² more accurately."""
        x = (4.0 * RNG.standard_normal(2048)).astype(np.float32)
        w = RNG.uniform(0.5, 1.5, 2048).astype(np.float32)
        ref = w * (x * np.float32(1.0 / np.sqrt(float(np.mean(x * x, dtype=np.float32)) + 1e-5)))
        assert_close(simd_ml_kernels.rms_norm(x, w), ref.astype(np.float64), rtol=2e-6, atol=self.ATOL,
                     label="RMSNorm vs NumPy float32")

    @pytest.mark.skipif(not HAS_TORCH or not hasattr(torch.nn.functional, "rms_norm"),
                        reason="PyTorch with F.rms_norm not installed")
    def test_rms_norm_vs_pytorch(self):
        x = RNG.standard_normal((4, 2048)).astype(np.float32)
        w = RNG.uniform(0.5, 1.5, 2048).astype(np.float32)
        ref = F.rms_norm(torch.from_numpy(x), (2048,), torch.from_numpy(w), 1e-5).numpy()
        assert_close(simd_ml_kernels.rms_norm(x, w), ref, rtol=2e-6, atol=self.ATOL,
                     label="vs PyTorch rms_norm")


class TestSiluMul:
    # Documented bound (silu_avx2.cpp): 1e-35*|up| + 1.1e-6*|ref| for finite inputs.
    RTOL = 1.1e-6

    def _check(self, g, u, label):
        out = simd_ml_kernels.silu_mul(g, u).astype(np.float64)
        with np.errstate(over="ignore"):
            ref = silu_numpy(g.astype(np.float64)) * u.astype(np.float64)
        over = np.abs(ref) > np.finfo(np.float32).max   # the float32 product overflows: ±inf
        assert (out[over] == np.sign(ref[over]) * np.inf).all(), f"{label}: overflow not ±inf"
        out, ref, u = out[~over], ref[~over], u[~over]
        err = np.abs(out - ref)
        bound = 1e-35 * np.abs(u.astype(np.float64)) + self.RTOL * np.abs(ref)
        worst = np.max(err / np.maximum(bound, 1e-300)) if err.size else 0.0
        assert (err <= bound).all(), f"{label}: worst error {worst:.3g}x the bound"

    @pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 15, 16, 17, 1023, 8192, 8193])
    def test_silu_mul_vs_float64(self, n):
        g = RNG.uniform(-200.0, 200.0, n).astype(np.float32)   # past the x < -88 cut-off
        u = (5.0 * RNG.standard_normal(n)).astype(np.float32)
        self._check(g, u, f"SiLU*up n={n}")

    def test_silu_mul_every_binade(self):
        e = np.arange(-126, 128)
        mags = np.concatenate([np.ldexp(m, e) for m in (1.0, 1.3, 1.9999999)])
        g = np.concatenate([mags, -mags]).astype(np.float32)
        u = RNG.uniform(-3.0, 3.0, g.size).astype(np.float32)
        self._check(g, u, "SiLU*up every binade")

    def test_silu_mul_equals_silu_times_up_bitwise(self):
        """One rounding after SiLU: the same bits as silu(g) * up on the same table."""
        g = RNG.uniform(-30.0, 30.0, 1027).astype(np.float32)
        u = RNG.standard_normal(1027).astype(np.float32)
        np.testing.assert_array_equal(simd_ml_kernels.silu_mul(g, u), simd_ml_kernels.silu(g) * u)

    def test_silu_mul_unaligned_and_2d(self):
        gb = RNG.uniform(-10, 10, 2 * 1024 + 1).astype(np.float32)
        ub = RNG.standard_normal(2 * 1024 + 1).astype(np.float32)
        g, u = gb[1:].reshape(2, 1024), ub[1:].reshape(2, 1024)
        out = simd_ml_kernels.silu_mul(g, u)
        assert out.shape == (2, 1024)
        self._check(g.ravel(), u.ravel(), "SiLU*up unaligned 2-D")

    def test_silu_mul_special_values(self):
        g = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0], np.float32)
        u = np.array([2.0, 2.0, 2.0, 5.0, np.nan], np.float32)
        out = simd_ml_kernels.silu_mul(g, u)
        assert out[0] == np.inf and np.isnan(out[1]) and np.isnan(out[2])
        assert out[3] == 0.0 and np.isnan(out[4])


# ─── Alignment / build_info Tests ─────────────────────────────────────────────

class TestBuildInfo:
    def test_build_info_returns_dict(self):
        info = simd_ml_kernels.build_info()
        assert isinstance(info, dict)
        assert "isa_compiled" in info or "ISA" in info

    def test_detected_isa_is_string(self):
        isa = simd_ml_kernels.detected_isa()
        assert isinstance(isa, str)
        assert any(kw in isa for kw in ("avx", "scalar", "sse"))

    def test_is_aligned_returns_bool(self):
        x = np.zeros(1024, dtype=np.float32)
        result = simd_ml_kernels.is_aligned(x)
        assert isinstance(result, bool)


if __name__ == "__main__":
    # python tests/test_precision.py write-q4_0-fixture <llama.cpp>/build/bin/libggml-base.so
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "write-q4_0-fixture":
        write_q4_0_fixture(sys.argv[2])
        print(f"wrote {Q4_0_FIXTURE}")
    else:
        sys.exit("usage: python tests/test_precision.py write-q4_0-fixture <libggml-base.so>")
