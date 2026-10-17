"""
tests/test_bindings_edge_cases.py — Edge case and error handling tests
for the Python bindings layer.

These tests verify that invalid inputs produce clear error messages rather
than segfaults or silent wrong results.
"""

import numpy as np
import pytest

try:
    import simd_ml_kernels
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    pytest.skip("simd_ml_kernels not built", allow_module_level=True)


class TestInputValidation:
    """Verify that invalid inputs are caught at the Python layer with clear errors."""

    def test_sgemm_wrong_dtype(self):
        A = np.random.randn(8, 8).astype(np.float64)   # wrong dtype
        B = np.random.randn(8, 8).astype(np.float32)
        with pytest.raises(RuntimeError, match="float32"):
            simd_ml_kernels.sgemm(A, B)

    def test_sgemm_wrong_ndim(self):
        A = np.random.randn(8).astype(np.float32)       # 1-D instead of 2-D
        B = np.random.randn(8, 8).astype(np.float32)
        with pytest.raises(RuntimeError):
            simd_ml_kernels.sgemm(A, B)

    def test_sgemm_k_mismatch(self):
        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(32, 8).astype(np.float32)   # K mismatch
        with pytest.raises(RuntimeError):
            simd_ml_kernels.sgemm(A, B)

    def test_gelu_wrong_dtype(self):
        x = np.random.randn(64).astype(np.float64)
        with pytest.raises(RuntimeError, match="float32"):
            simd_ml_kernels.gelu(x)

    def test_gelu_inplace_read_only(self):
        x = np.zeros(64, dtype=np.float32)
        x.flags.writeable = False
        with pytest.raises(RuntimeError, match="writeable"):
            simd_ml_kernels.gelu_inplace(x)

    def test_layer_norm_gamma_wrong_size(self):
        x     = np.random.randn(64).astype(np.float32)
        gamma = np.ones(32, dtype=np.float32)            # wrong size
        with pytest.raises(RuntimeError):
            simd_ml_kernels.layer_norm(x, gamma=gamma)

    def test_softmax_wrong_dtype(self):
        x = np.random.randn(64).astype(np.float64)
        with pytest.raises(RuntimeError, match="float32"):
            simd_ml_kernels.softmax(x)

    def test_softmax_unsupported_axis(self):
        x = np.random.randn(8, 16, 32).astype(np.float32)
        with pytest.raises(RuntimeError, match="last axis"):
            simd_ml_kernels.softmax(x, axis=0)


class TestEmptyAndSingleElement:
    """Verify correct behaviour for boundary sizes."""

    def test_sgemm_1x1(self):
        A = np.array([[3.0]], dtype=np.float32)
        B = np.array([[4.0]], dtype=np.float32)
        C = simd_ml_kernels.sgemm(A, B)
        assert abs(float(C[0, 0]) - 12.0) < 1e-4

    def test_gelu_single_element(self):
        x = np.array([0.0], dtype=np.float32)
        out = simd_ml_kernels.gelu(x)
        # GeLU(0) = 0
        assert abs(float(out[0])) < 1e-6

    def test_relu_single_negative(self):
        x = np.array([-5.0], dtype=np.float32)
        assert float(simd_ml_kernels.relu(x)[0]) == 0.0

    def test_softmax_single_element(self):
        x = np.array([42.0], dtype=np.float32)
        out = simd_ml_kernels.softmax(x)
        assert abs(float(out[0]) - 1.0) < 1e-6

    def test_layer_norm_single_element(self):
        # LayerNorm of a single element: (x - x) / sqrt(0 + eps) = 0
        x = np.array([5.0], dtype=np.float32)
        out = simd_ml_kernels.layer_norm(x)
        assert abs(float(out[0])) < 1e-3


class TestNumericalEdgeCases:
    """Verify behaviour at numerical boundaries."""

    def test_sgemm_identity(self):
        """Multiplying by identity should return the original matrix."""
        n = 32
        A = np.eye(n, dtype=np.float32) * 2.0
        B = np.eye(n, dtype=np.float32)
        C = simd_ml_kernels.sgemm(A, B)
        assert np.allclose(C, A, atol=1e-5)

    def test_gelu_large_positive(self):
        """GeLU(x) ≈ x for large x."""
        x = np.array([20.0, 50.0, 100.0], dtype=np.float32)
        out = simd_ml_kernels.gelu(x)
        assert np.allclose(out, x, rtol=1e-3)

    def test_gelu_large_negative(self):
        """GeLU(x) ≈ 0 for large negative x."""
        x = np.array([-20.0, -50.0, -100.0], dtype=np.float32)
        out = simd_ml_kernels.gelu(x)
        assert np.allclose(out, 0.0, atol=1e-5)

    def test_softmax_uniform_input(self):
        """softmax of identical values should produce uniform distribution."""
        n = 64
        x = np.ones(n, dtype=np.float32)
        out = simd_ml_kernels.softmax(x)
        expected = 1.0 / n
        assert np.allclose(out, expected, atol=1e-6)

    def test_layer_norm_constant_input(self):
        """LayerNorm of constant input: output should be all zeros (var=0 → normalize by eps)."""
        x = np.full(64, 3.14159, dtype=np.float32)
        out = simd_ml_kernels.layer_norm(x)
        # (x - mean) / sqrt(0 + eps) = 0 / sqrt(eps) = 0
        assert np.allclose(out, 0.0, atol=1e-4)


class TestGEMMConfig:
    """Tests for the GEMMConfig callable configuration object (roadmap §9)."""

    def test_default_construction(self):
        gemm = simd_ml_kernels.GEMMConfig()
        assert gemm.alpha == 1.0
        assert gemm.beta == 0.0
        assert gemm.isa == ""

    def test_custom_construction(self):
        gemm = simd_ml_kernels.GEMMConfig(alpha=2.0, beta=0.5, isa="avx2")
        assert gemm.alpha == 2.0
        assert gemm.beta == 0.5
        assert gemm.isa == "avx2"

    def test_call_matches_sgemm(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((32, 64)).astype(np.float32)
        B = rng.standard_normal((64, 16)).astype(np.float32)

        gemm = simd_ml_kernels.GEMMConfig()
        C1 = gemm(A, B)
        C2 = simd_ml_kernels.sgemm(A, B)
        assert np.allclose(C1, C2, rtol=1e-4)

    def test_stored_alpha_beta_applied(self):
        rng = np.random.default_rng(8)
        A = rng.standard_normal((16, 16)).astype(np.float32)
        B = rng.standard_normal((16, 16)).astype(np.float32)
        C_init = np.ones((16, 16), dtype=np.float32)

        gemm = simd_ml_kernels.GEMMConfig(alpha=2.0, beta=0.5)
        C = gemm(A, B, C=C_init.copy())

        ref = 2.0 * (A.astype(np.float64) @ B.astype(np.float64)) + 0.5 * 1.0
        assert np.allclose(C.astype(np.float64), ref, rtol=1e-3, atol=1e-4)

    def test_per_call_override(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((16, 16)).astype(np.float32)
        B = rng.standard_normal((16, 16)).astype(np.float32)

        gemm = simd_ml_kernels.GEMMConfig(alpha=2.0, beta=0.5)
        # Override alpha/beta for this call only
        C = gemm(A, B, alpha=1.0, beta=0.0)
        ref = (A.astype(np.float64) @ B.astype(np.float64))
        assert np.allclose(C.astype(np.float64), ref, rtol=1e-3)

        # Stored config is unaffected by the override
        assert gemm.alpha == 2.0
        assert gemm.beta == 0.5

    def test_invalid_isa_raises(self):
        with pytest.raises(ValueError, match="isa must be one of"):
            simd_ml_kernels.GEMMConfig(isa="cuda")

    def test_invalid_tile_size_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            simd_ml_kernels.GEMMConfig(tile_m=-1)
        with pytest.raises(ValueError, match="must be positive"):
            simd_ml_kernels.GEMMConfig(tile_n=0)

    def test_repr_shows_non_default_values(self):
        gemm = simd_ml_kernels.GEMMConfig(alpha=2.0, isa="avx2")
        r = repr(gemm)
        assert "alpha=2" in r
        assert "isa='avx2'" in r

    def test_tile_params_accepted(self):
        # Tile params are informational but must not raise for valid positive values
        gemm = simd_ml_kernels.GEMMConfig(tile_m=64, tile_n=64, tile_k=128, mr=6, nr=16)
        assert gemm.tile_m == 64
        assert gemm.mr == 6


class TestSgemmIsaKwarg:
    """Tests for the isa= keyword argument on sgemm() — genuine per-call
    kernel forcing (not just accepted-and-ignored), see ROADMAP.md §v0.8."""

    def test_auto_and_simd_isa_values_accepted(self):
        rng = np.random.default_rng(10)
        A = rng.standard_normal((16, 16)).astype(np.float32)
        B = rng.standard_normal((16, 16)).astype(np.float32)
        for isa in ["", "avx2", "avx512"]:
            if isa == "avx512" and not simd_ml_kernels.avx512_available():
                continue  # would correctly raise on non-AVX-512 hardware
            C = simd_ml_kernels.sgemm(A, B, isa=isa)
            assert C.shape == (16, 16)

    def test_scalar_isa_raises_not_implemented(self):
        """isa='scalar' is a recognized value but not yet wired to a forced
        full-matrix scalar path — must raise a clear error, not silently
        fall back to auto (which would be a transparency violation: the
        user explicitly asked for scalar and got SIMD instead, unannounced).
        """
        rng = np.random.default_rng(12)
        A = rng.standard_normal((8, 8)).astype(np.float32)
        B = rng.standard_normal((8, 8)).astype(np.float32)
        with pytest.raises(RuntimeError, match="not yet implemented"):
            simd_ml_kernels.sgemm(A, B, isa="scalar")

    def test_invalid_isa_raises(self):
        rng = np.random.default_rng(11)
        A = rng.standard_normal((8, 8)).astype(np.float32)
        B = rng.standard_normal((8, 8)).astype(np.float32)
        with pytest.raises(RuntimeError, match="isa must be one of"):
            simd_ml_kernels.sgemm(A, B, isa="cuda")

    def test_avx512_requested_on_unsupported_hardware_raises(self):
        """Cannot directly simulate missing AVX-512 hardware in-process, so
        this test only runs its assertion when avx512_available() is False
        (e.g. on a CI runner without AVX-512) — otherwise it's a no-op,
        documenting the expected behavior for future readers either way.
        """
        rng = np.random.default_rng(13)
        A = rng.standard_normal((8, 8)).astype(np.float32)
        B = rng.standard_normal((8, 8)).astype(np.float32)
        if not simd_ml_kernels.avx512_available():
            with pytest.raises(RuntimeError, match="does not support"):
                simd_ml_kernels.sgemm(A, B, isa="avx512")

    def test_forced_avx2_and_avx512_agree_numerically(self):
        """Both kernels must independently compute the same GEMM correctly —
        this is a genuine cross-check between two different code paths, not
        just each against a reference (see also the C++-level
        test_gemm_isa_override in tests/test_gemm_packed.cpp for a direct,
        non-numerical proof that the override actually changes dispatch).
        """
        if not simd_ml_kernels.avx512_available():
            pytest.skip("AVX-512 not available on this machine")
        rng = np.random.default_rng(14)
        A = rng.standard_normal((96, 128)).astype(np.float32)
        B = rng.standard_normal((128, 96)).astype(np.float32)
        C_avx2 = simd_ml_kernels.sgemm(A, B, isa="avx2")
        C_avx512 = simd_ml_kernels.sgemm(A, B, isa="avx512")
        assert np.allclose(C_avx2, C_avx512, rtol=1e-4, atol=1e-4)

    def test_isa_override_does_not_leak_to_subsequent_calls(self):
        """A forced isa= on one call must not affect later calls that don't
        specify isa= — this is the per-call (not sticky/global) contract.
        """
        rng = np.random.default_rng(15)
        A = rng.standard_normal((32, 32)).astype(np.float32)
        B = rng.standard_normal((32, 32)).astype(np.float32)

        C_before = simd_ml_kernels.sgemm(A, B)          # auto, establishes baseline
        simd_ml_kernels.sgemm(A, B, isa="avx2")           # forced call, should not leak
        C_after = simd_ml_kernels.sgemm(A, B)            # auto again

        assert np.array_equal(C_before, C_after), \
            "isa= override leaked into a subsequent auto-detect call"

    def test_gemmconfig_isa_forces_kernel_too(self):
        """GEMMConfig(isa=...) delegates to sgemm's isa= — verify it's
        actually wired through, not just stored and ignored.
        """
        rng = np.random.default_rng(16)
        A = rng.standard_normal((32, 32)).astype(np.float32)
        B = rng.standard_normal((32, 32)).astype(np.float32)

        gemm_scalar_request = simd_ml_kernels.GEMMConfig(isa="scalar")
        with pytest.raises(RuntimeError, match="not yet implemented"):
            gemm_scalar_request(A, B)


class TestAvx512Available:
    """Tests for the avx512_available() hardware-capability diagnostic."""

    def test_returns_bool(self):
        result = simd_ml_kernels.avx512_available()
        assert isinstance(result, bool)

    def test_consistent_with_detected_isa(self):
        """If detected_isa() reports 'avx512', avx512_available() must be
        True — these two diagnostics must never contradict each other."""
        if simd_ml_kernels.detected_isa() == "avx512":
            assert simd_ml_kernels.avx512_available() is True


class TestDeprecatedImportName:
    """`import simd_kernels` (the pre-rename name) keeps working until 0.5.0."""

    def test_old_name_warns_and_is_the_same_module(self):
        import importlib
        import sys

        sys.modules.pop("simd_kernels", None)
        with pytest.warns(DeprecationWarning, match="simd_ml_kernels"):
            old = importlib.import_module("simd_kernels")
        assert old is simd_ml_kernels
        assert old.sgemm is simd_ml_kernels.sgemm
