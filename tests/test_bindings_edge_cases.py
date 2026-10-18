"""
tests/test_bindings_edge_cases.py — Edge case and error handling tests
for the Python bindings layer.

These tests verify that invalid inputs produce clear error messages rather
than segfaults or silent wrong results.
"""

import os

import numpy as np
import pytest

try:
    import simd_ml_kernels
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    pytest.skip("simd_ml_kernels not built", allow_module_level=True)


class TestInputValidation:
    """Invalid inputs raise DTypeError (TypeError) or ArgumentError (ValueError)
    with an actionable message; both also subclass RuntimeError until 0.5.0."""

    def test_sgemm_wrong_dtype(self):
        A = np.random.randn(8, 8).astype(np.float64)   # wrong dtype
        B = np.random.randn(8, 8).astype(np.float32)
        with pytest.raises(simd_ml_kernels.DTypeError, match="float32"):
            simd_ml_kernels.sgemm(A, B)

    def test_sgemm_wrong_ndim(self):
        A = np.random.randn(8).astype(np.float32)       # 1-D instead of 2-D
        B = np.random.randn(8, 8).astype(np.float32)
        with pytest.raises(simd_ml_kernels.ArgumentError, match="2-D"):
            simd_ml_kernels.sgemm(A, B)

    def test_sgemm_k_mismatch(self):
        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(32, 8).astype(np.float32)   # K mismatch
        with pytest.raises(simd_ml_kernels.ArgumentError, match="Shape mismatch"):
            simd_ml_kernels.sgemm(A, B)

    def test_sgemm_k_mismatch_transposed_names_the_transpose(self):
        A = np.ones((16, 8), np.float32)    # A.T is 8x16
        B = np.ones((8, 32), np.float32)    # B.T is 32x8: K 16 vs 32
        with pytest.raises(simd_ml_kernels.ArgumentError, match=r"A\.T is \[8×16\].*B\.T is \[32×8\]"):
            simd_ml_kernels.sgemm(A, B, trans_a=True, trans_b=True)

    def test_sgemm_trans_flags_are_keyword_only(self):
        A = np.ones((4, 4), np.float32)
        with pytest.raises(TypeError):
            simd_ml_kernels.sgemm(A, A, None, 1.0, 0.0, "", True)

    def test_sgemm_trans_c_shape_checked(self):
        A = np.ones((8, 4), np.float32)     # trans_a: op(A) is 4x8
        B = np.ones((8, 6), np.float32)
        with pytest.raises(simd_ml_kernels.ArgumentError):
            simd_ml_kernels.sgemm(A, B, np.zeros((8, 6), np.float32), trans_a=True)

    def test_gelu_wrong_dtype(self):
        x = np.random.randn(64).astype(np.float64)
        with pytest.raises(simd_ml_kernels.DTypeError, match="float32"):
            simd_ml_kernels.gelu(x)

    def test_gelu_inplace_read_only(self):
        x = np.zeros(64, dtype=np.float32)
        x.flags.writeable = False
        with pytest.raises(simd_ml_kernels.ArgumentError, match="writeable"):
            simd_ml_kernels.gelu_inplace(x)

    def test_layer_norm_gamma_wrong_size(self):
        x     = np.random.randn(64).astype(np.float32)
        gamma = np.ones(32, dtype=np.float32)            # wrong size
        with pytest.raises(simd_ml_kernels.ArgumentError, match="gamma"):
            simd_ml_kernels.layer_norm(x, gamma=gamma)

    def test_softmax_wrong_dtype(self):
        x = np.random.randn(64).astype(np.float64)
        with pytest.raises(simd_ml_kernels.DTypeError, match="float32"):
            simd_ml_kernels.softmax(x)

    def test_softmax_unsupported_axis(self):
        x = np.random.randn(8, 16, 32).astype(np.float32)
        with pytest.raises(simd_ml_kernels.ArgumentError, match="last axis"):
            simd_ml_kernels.softmax(x, axis=0)

    def test_error_classes_have_builtin_and_legacy_bases(self):
        assert issubclass(simd_ml_kernels.DTypeError, TypeError)
        assert issubclass(simd_ml_kernels.ArgumentError, ValueError)
        # Code written for <= 0.2 caught RuntimeError; that keeps working.
        with pytest.raises(RuntimeError):
            simd_ml_kernels.sgemm(np.ones((2, 2)), np.ones((2, 2), np.float32))
        with pytest.raises(RuntimeError):
            simd_ml_kernels.sgemm(np.ones((2, 3), np.float32), np.ones((2, 2), np.float32))


def _bf16_bits(x):
    """float32 -> BF16 bit patterns (truncation), as uint16."""
    return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


class TestOutputArgumentIsNeverCopied:
    """C is written in place; anything that would require converting it is
    rejected, and the caller's array is left untouched. Before this check, a
    float64 or strided C was silently copied and the result was lost."""

    @staticmethod
    def _call(fn, C):
        if fn == "sgemm":
            A = np.ones((4, 3), np.float32)
            B = np.ones((3, 5), np.float32)
            return simd_ml_kernels.sgemm(A, B, C=C)
        if fn == "sgemm_f16":
            if not simd_ml_kernels.f16c_available():
                pytest.skip("F16C not available")
            A = np.ones((4, 3), np.float16)
            B = np.ones((3, 5), np.float16)
            return simd_ml_kernels.sgemm_f16(A, B, C=C)
        A = _bf16_bits(np.ones((4, 3)))
        B = _bf16_bits(np.ones((3, 5)))
        return simd_ml_kernels.sgemm_bf16(A, B, C=C)

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    def test_float64_C_rejected_and_untouched(self, fn):
        C = np.zeros((4, 5), dtype=np.float64)
        with pytest.raises(simd_ml_kernels.DTypeError, match="C must be float32"):
            self._call(fn, C)
        assert not C.any()

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    def test_non_contiguous_C_rejected_and_untouched(self, fn):
        C = np.zeros((4, 5), dtype=np.float32, order="F")
        with pytest.raises(simd_ml_kernels.ArgumentError, match="C-contiguous"):
            self._call(fn, C)
        assert not C.any()

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    def test_read_only_C_rejected(self, fn):
        C = np.zeros((4, 5), dtype=np.float32)
        C.flags.writeable = False
        with pytest.raises(simd_ml_kernels.ArgumentError, match="writeable"):
            self._call(fn, C)

    def test_non_array_C_rejected(self):
        with pytest.raises(simd_ml_kernels.DTypeError, match="numpy.ndarray"):
            self._call("sgemm", [[0.0] * 5] * 4)

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    def test_valid_C_is_written_in_place(self, fn):
        C = np.zeros((4, 5), dtype=np.float32)
        out = self._call(fn, C)
        assert out is C or np.shares_memory(out, C)
        np.testing.assert_array_equal(C, np.full((4, 5), 3.0, np.float32))


class TestSgemmBf16Inputs:
    def test_uint16_bit_patterns_match_reference(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((9, 33)).astype(np.float32)
        B = rng.standard_normal((33, 17)).astype(np.float32)
        Ab, Bb = _bf16_bits(A), _bf16_bits(B)
        # Reference on the BF16-rounded values in float64. Products of two BF16
        # values are exact in FP32, so the only error is FP32 accumulation.
        Ar = (Ab.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        Br = (Bb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        np.testing.assert_allclose(simd_ml_kernels.sgemm_bf16(Ab, Bb), Ar @ Br,
                                   rtol=1e-5, atol=1e-5)

    def test_float16_rejected(self):
        A = np.ones((4, 4), np.float16)
        with pytest.raises(simd_ml_kernels.DTypeError, match="sgemm_f16"):
            simd_ml_kernels.sgemm_bf16(A, A)

    @pytest.mark.parametrize("dtype", [np.float32, np.int16, np.uint8])
    def test_other_dtypes_rejected(self, dtype):
        A = np.ones((4, 4), dtype)
        with pytest.raises(simd_ml_kernels.DTypeError, match="BF16"):
            simd_ml_kernels.sgemm_bf16(A, A)

    def test_ml_dtypes_bfloat16_accepted(self):
        ml_dtypes = pytest.importorskip("ml_dtypes")
        A = np.arange(12, dtype=np.float32).reshape(3, 4)
        B = np.ones((4, 2), np.float32)
        out = simd_ml_kernels.sgemm_bf16(A.astype(ml_dtypes.bfloat16),
                                         B.astype(ml_dtypes.bfloat16))
        np.testing.assert_allclose(out, A @ B, rtol=1e-2)


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
    """sgemm(isa=...) runs that kernel table for this call only; SIMD_ML_ISA
    caps the table for the whole process (src/dispatch/kernels.hpp)."""

    @staticmethod
    def usable():
        return simd_ml_kernels.build_info()["isa_usable"].split()

    @pytest.mark.parametrize("shape", [(1, 1, 1), (7, 9, 5), (33, 17, 65), (130, 129, 131)])
    def test_every_usable_isa_matches_float64(self, shape):
        # Shapes cover the tails of 8/16-wide tiles, the unpacked path
        # (max <= 128) and the packed path. Bound: |C - AB| <= 2*K*u*|A||B|.
        m, k, n = shape
        rng = np.random.default_rng(10)
        A = rng.standard_normal((m, k)).astype(np.float32)
        B = rng.standard_normal((k, n)).astype(np.float32)
        a64, b64 = A.astype(np.float64), B.astype(np.float64)
        bound = 2 * k * 2.0 ** -24 * (np.abs(a64) @ np.abs(b64))
        assert "scalar" in self.usable()
        for isa in self.usable():
            err = np.abs(simd_ml_kernels.sgemm(A, B, isa=isa) - a64 @ b64)
            assert np.all(err <= bound), f"isa={isa}: max err/bound {np.max(err / bound):.3f}"

    def test_compiled_but_unusable_isa_raises(self):
        unusable = sorted(set(simd_ml_kernels.build_info()["isa_compiled"].split()) - set(self.usable()))
        if not unusable:
            pytest.skip("this CPU runs every ISA compiled into the build")
        A = np.ones((8, 8), np.float32)
        for isa in unusable:
            with pytest.raises(RuntimeError, match="cannot run it"):
                simd_ml_kernels.sgemm(A, A, isa=isa)

    def test_invalid_isa_raises(self):
        A = np.ones((8, 8), np.float32)
        with pytest.raises(simd_ml_kernels.ArgumentError, match="isa must be one of"):
            simd_ml_kernels.sgemm(A, A, isa="cuda")

    def test_isa_override_does_not_leak_to_subsequent_calls(self):
        rng = np.random.default_rng(15)
        A = rng.standard_normal((32, 32)).astype(np.float32)
        B = rng.standard_normal((32, 32)).astype(np.float32)
        C_before = simd_ml_kernels.sgemm(A, B)
        simd_ml_kernels.sgemm(A, B, isa="scalar")
        assert np.array_equal(C_before, simd_ml_kernels.sgemm(A, B))
        assert simd_ml_kernels.detected_isa() == self.usable()[-1] or "SIMD_ML_ISA" in os.environ

    def test_gemmconfig_isa_forces_kernel_too(self):
        rng = np.random.default_rng(16)
        A = rng.standard_normal((40, 33)).astype(np.float32)
        B = rng.standard_normal((33, 21)).astype(np.float32)
        for isa in self.usable():
            assert np.array_equal(simd_ml_kernels.GEMMConfig(isa=isa)(A, B),
                                  simd_ml_kernels.sgemm(A, B, isa=isa))

    @pytest.mark.parametrize("cap", ["scalar", "avx2", "avx512"])
    def test_simd_ml_isa_env_caps_the_table(self, cap):
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-c", "import simd_ml_kernels as s; print(s.detected_isa())"],
            env=dict(os.environ, SIMD_ML_ISA=cap), capture_output=True, text=True, check=True)
        order = ["scalar", "avx2", "avx512"]
        usable = self.usable()
        expect = [i for i in order[: order.index(cap) + 1] if i in usable][-1]
        assert out.stdout.strip() == expect


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


class TestDefaultTableMatchesCpu:
    """With SIMD_ML_ISA unset, the active table is the widest one the CPU's
    flags (from the OS, /proc/cpuinfo) allow. On an AVX-512 CI runner this
    fails if the AVX-512 path is not taken."""

    def test_default_table_matches_cpuinfo(self):
        import subprocess
        import sys

        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                flags = next(line for line in f if line.startswith("flags")).split()
        except (OSError, StopIteration):
            pytest.skip("no /proc/cpuinfo")
        has = set(flags)
        compiled = simd_ml_kernels.build_info()["isa_compiled"].split()
        expect = "scalar"
        if "avx2" in compiled and {"avx2", "fma", "f16c"} <= has:
            expect = "avx2"
            if "avx512" in compiled and {"avx512f", "avx512dq", "avx512bw", "avx512vl"} <= has:
                expect = "avx512"
        env = {k: v for k, v in os.environ.items() if k != "SIMD_ML_ISA"}
        out = subprocess.run(
            [sys.executable, "-c", "import simd_ml_kernels as s; print(s.detected_isa())"],
            env=env, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == expect, f"CPU flags allow {expect}, got {out.stdout.strip()}"


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


class TestSizeZero:
    """Empty dimensions follow BLAS/NumPy: an empty K-sum gives C = beta*C
    (beta == 0 overwrites, even NaN), and empty arrays come back empty.
    Before this was fixed, sgemm with K == 0 returned uninitialised memory and
    layer_norm on an empty last axis crashed the interpreter."""

    @staticmethod
    def _gemm(fn, M, K, N, **kw):
        if fn == "sgemm":
            return simd_ml_kernels.sgemm(np.ones((M, K), np.float32), np.ones((K, N), np.float32), **kw)
        if fn == "sgemm_f16":
            if not simd_ml_kernels.f16c_available():
                pytest.skip("F16C not available")
            return simd_ml_kernels.sgemm_f16(np.ones((M, K), np.float16), np.ones((K, N), np.float16), **kw)
        return simd_ml_kernels.sgemm_bf16(_bf16_bits(np.ones((M, K))), _bf16_bits(np.ones((K, N))), **kw)

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    @pytest.mark.parametrize("M", [3, 300])   # 300 > 128 exercises the packed path
    def test_k_zero_allocated_output_is_zero(self, fn, M):
        for _ in range(3):  # fresh allocations; garbage used to show up here
            C = self._gemm(fn, M, 0, 5)
            assert C.shape == (M, 5)
            np.testing.assert_array_equal(C, 0.0)

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    def test_k_zero_beta_zero_overwrites_nan(self, fn):
        C = np.full((4, 5), np.nan, np.float32)
        self._gemm(fn, 4, 0, 5, C=C, beta=0.0)
        np.testing.assert_array_equal(C, 0.0)

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    def test_k_zero_scales_by_beta(self, fn):
        C = np.full((4, 5), 3.0, np.float32)
        self._gemm(fn, 4, 0, 5, C=C, beta=0.5)
        np.testing.assert_array_equal(C, 1.5)

    @pytest.mark.parametrize("fn", ["sgemm", "sgemm_f16", "sgemm_bf16"])
    @pytest.mark.parametrize("M,N", [(0, 5), (4, 0), (0, 0)])
    def test_empty_output(self, fn, M, N):
        assert self._gemm(fn, M, 3, N).shape == (M, N)

    @pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 5)])
    def test_layer_norm_empty(self, shape):
        assert simd_ml_kernels.layer_norm(np.ones(shape, np.float32)).shape == shape

    @pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 5)])
    def test_softmax_empty(self, shape):
        assert simd_ml_kernels.softmax(np.ones(shape, np.float32)).shape == shape

    @pytest.mark.parametrize("name", ["gelu", "relu", "silu"])
    def test_elementwise_empty(self, name):
        assert getattr(simd_ml_kernels, name)(np.ones((0, 4), np.float32)).shape == (0, 4)


class TestThreadControl:
    def test_build_info_reports_openmp(self):
        assert isinstance(simd_ml_kernels.build_info()["openmp"], bool)

    def test_set_num_threads_without_openmp_warns(self):
        if simd_ml_kernels.build_info()["openmp"]:
            pytest.skip("built with OpenMP")
        with pytest.warns(RuntimeWarning, match="without OpenMP"):
            simd_ml_kernels.set_num_threads(4)
        assert simd_ml_kernels.get_num_threads() == 1
        simd_ml_kernels.set_num_threads(1)  # n == 1 is valid and silent

    def test_set_num_threads_with_openmp_takes_effect(self):
        if not simd_ml_kernels.build_info()["openmp"]:
            pytest.skip("built without OpenMP")
        simd_ml_kernels.set_num_threads(2)
        assert simd_ml_kernels.get_num_threads() == 2
        simd_ml_kernels.set_num_threads(1)

    @pytest.mark.parametrize("omp_num_threads", ["1", "3"])
    def test_default_thread_count_follows_omp_num_threads(self, omp_num_threads):
        """Until set_num_threads is called, GEMM uses OpenMP's default count."""
        import subprocess
        import sys
        env = dict(os.environ, OMP_NUM_THREADS=omp_num_threads)
        out = subprocess.run([sys.executable, "-c",
                              "import simd_ml_kernels as s; print(s.get_num_threads())"],
                             env=env, capture_output=True, text=True, check=True).stdout
        expected = omp_num_threads if simd_ml_kernels.build_info()["openmp"] else "1"
        assert out.strip() == expected
