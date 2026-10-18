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

import numpy as np
import pytest

try:
    import simd_ml_kernels
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
    ])
    def test_sgemm_vs_numpy(self, M, N, K):
        A = rand_matrix(M, K)
        B = rand_matrix(K, N)
        C = np.zeros((M, N), dtype=np.float32)

        simd_ml_kernels.sgemm(A, B, C)
        C_ref = A.astype(np.float64) @ B.astype(np.float64)

        assert_close(C, C_ref, rtol=self.GEMM_TOL, atol=self.GEMM_ATOL,
                    label=f"GEMM [{M}×{N}×{K}]")

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


# ─── GeLU Tests ──────────────────────────────────────────────────────────────

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
