/**
 * pybind_entry.cpp — Python Bindings via pybind11
 *
 * Exposes SIMD GEMM and activation kernels to Python with NumPy-compatible
 * array access. The binding layer handles:
 *   - shape, dtype, and contiguity validation
 *   - GIL release during computation (allows Python multithreading)
 *   - Aligned temporary buffers when needed
 *   - Clear error messages for common user mistakes
 *
 * ─── Zero-Copy Strategy ───────────────────────────────────────────────────────
 * pybind11's py::array_t<float, py::array::c_contiguous> descriptor enforces:
 *   1. dtype == float32
 *   2. C-contiguous memory layout (row-major)
 *   3. Direct access to the buffer pointer via .mutable_data() / .data()
 *
 * This means `np.array(..., dtype=np.float32)` passed from Python goes directly
 * to the C++ kernel with no allocation or copy overhead.
 *
 * ─── Build ────────────────────────────────────────────────────────────────────
 *   pip install -e .       # via scikit-build-core (recommended)
 *
 * ─── Python Usage ─────────────────────────────────────────────────────────────
 *   import numpy as np
 *   import simd_ml_kernels
 *
 *   A = np.random.randn(256, 256).astype(np.float32)
 *   B = np.random.randn(256, 256).astype(np.float32)
 *   C = np.zeros((256, 256), dtype=np.float32)
 *   simd_ml_kernels.sgemm(A, B, C)
 *
 *   x = np.random.randn(65536).astype(np.float32)
 *   y = simd_ml_kernels.gelu(x)
 *   simd_ml_kernels.gelu_inplace(x)
 *
 *   print(simd_ml_kernels.build_info())
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <utility>

#include "../kernels/activations/activations.hpp"
#include "../kernels/gemm/avx2_gemm_packed.hpp"
#include "../kernels/gemm/f16_gemm.hpp"
#include "../kernels/gemm/bf16_gemm.hpp"
#include "../kernels/intrinsic_gelu.hpp"
#include "../kernels/cache_alloc.hpp"
#include "../dispatch/kernel_registry.hpp"

namespace py = pybind11;

// ─── Argument errors ──────────────────────────────────────────────────────────
// Registered in the module as simd_ml_kernels.DTypeError (bases TypeError,
// RuntimeError) and simd_ml_kernels.ArgumentError (bases ValueError,
// RuntimeError). Before 0.3 every argument error was a plain RuntimeError;
// the extra RuntimeError base keeps `except RuntimeError` working until 0.5.0.
struct DTypeError    : std::runtime_error { using std::runtime_error::runtime_error; };
struct ArgumentError : std::runtime_error { using std::runtime_error::runtime_error; };

// ─── Validation helpers ───────────────────────────────────────────────────────

static void check_float32(const py::array& arr, const char* name) {
    if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != 4) {
        throw DTypeError(
            std::string(name) + " must be float32 (dtype=np.float32). "
            "Call arr.astype(np.float32) before passing.");
    }
}

static void check_contiguous(const py::array& arr, const char* name) {
    if (!arr.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw ArgumentError(
            std::string(name) + " must be C-contiguous. "
            "Call numpy.ascontiguousarray(arr) before passing.");
    }
}

static void validate_array(const py::array& arr, const char* name) {
    check_float32(arr, name);
    check_contiguous(arr, name);
}

static void check_writeable(const py::array& arr, const char* name) {
    if (!arr.attr("flags").attr("writeable").cast<bool>()) {
        throw ArgumentError(
            std::string(name) + " must be writeable. "
            "The array may be read-only; make a copy first.");
    }
}

/**
 * output_matrix — validate a caller-supplied output C without converting it.
 *
 * C is written in place, so it must already be exactly what the kernel
 * writes: a float32, C-contiguous, writeable [M, N] ndarray. Casting it to
 * py::array_t<float> (the previous code) silently made a converted copy for a
 * float64 or strided C, wrote the result into that copy and left the caller's
 * array unchanged.
 */
static py::array_t<float> output_matrix(const py::object& C, py::ssize_t M, py::ssize_t N) {
    if (!py::isinstance<py::array>(C))
        throw DTypeError("C must be a numpy.ndarray of dtype float32 (or None)");
    auto arr = py::reinterpret_borrow<py::array>(C);
    validate_array(arr, "C");
    check_writeable(arr, "C");
    if (arr.ndim() != 2 || arr.shape(0) != M || arr.shape(1) != N)
        throw ArgumentError("C must have shape [" + std::to_string(M) + "×" +
                            std::to_string(N) + "]");
    return py::reinterpret_borrow<py::array_t<float>>(arr);
}

// ─── Binding: sgemm ──────────────────────────────────────────────────────────
/**
 * sgemm(A, B, C=None, alpha=1.0, beta=0.0) -> np.ndarray
 *
 * Computes C = alpha * A @ B + beta * C  (in-place if C provided, else allocates).
 *
 * Args:
 *   A     : np.ndarray shape [M, K], dtype float32, C-contiguous
 *   B     : np.ndarray shape [K, N], dtype float32, C-contiguous
 *   C     : np.ndarray shape [M, N], dtype float32, C-contiguous, writeable (optional)
 *   alpha : float scalar (default 1.0)
 *   beta  : float scalar for C scaling (default 0.0 = overwrite)
 *
 * Returns:
 *   C array (same object as input C if provided, else newly allocated)
 */

/**
 * IsaOverrideGuard — RAII wrapper around simd_ml::gemm::set_gemm_isa_override.
 *
 * Saves the current thread-local override on construction, applies the
 * requested one, and restores the previous value on destruction (including
 * on the exception path) — so a single sgemm(..., isa=...) call cannot leak
 * its override into unrelated subsequent calls on the same thread.
 */
class IsaOverrideGuard {
public:
    explicit IsaOverrideGuard(const std::string& isa)
        : previous_(simd_ml::gemm::get_gemm_isa_override()) {
        simd_ml::gemm::set_gemm_isa_override(isa.empty() ? nullptr : isa.c_str());
    }
    ~IsaOverrideGuard() {
        simd_ml::gemm::set_gemm_isa_override(previous_.c_str());
    }
    IsaOverrideGuard(const IsaOverrideGuard&) = delete;
    IsaOverrideGuard& operator=(const IsaOverrideGuard&) = delete;
private:
    std::string previous_;
};

static py::array_t<float> py_sgemm(py::array A,
                                     py::array B,
                                     py::object C   = py::none(),
                                     float alpha    = 1.0f,
                                     float beta     = 0.0f,
                                     const std::string& isa = "") {
    // isa="" (default) uses the runtime auto-detected kernel — this is what
    // essentially all callers want. Explicit values force a specific kernel
    // for exactly this call (see IsaOverrideGuard above for the per-call,
    // not-sticky, semantics).
    if (!isa.empty() && isa != "avx2" && isa != "avx512" && isa != "scalar") {
        throw ArgumentError(
            "isa must be one of: 'avx2', 'avx512', 'scalar' (got '" + isa + "')");
    }
    if (isa == "scalar") {
        // Honest error rather than silently falling back to auto: a forced
        // full-matrix scalar GEMM path is not implemented for sgemm_packed
        // (scalar code only runs internally for small edge/tail blocks).
        throw std::runtime_error(
            "isa='scalar' is not yet implemented for sgemm_packed "
            "(scalar_sgemm() is available separately as a reference "
            "implementation, but is not wired into sgemm's isa= override). "
            "Use isa='avx2' or isa='avx512' to force a SIMD kernel, or "
            "isa='' for automatic hardware detection.");
    }
    if (isa == "avx512" && !simd_ml::gemm::gemm_packed_avx512_hardware_available()) {
        throw std::runtime_error(
            "isa='avx512' was requested but this CPU does not support "
            "AVX-512F+DQ. Use isa='avx2' or isa='' (auto-detect) instead. "
            "Check simd_ml_kernels.avx512_available() to query hardware support.");
    }

    validate_array(A, "A");
    validate_array(B, "B");
    if (A.ndim() != 2) throw ArgumentError("A must be 2-D");
    if (B.ndim() != 2) throw ArgumentError("B must be 2-D");

    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));

    if (B.shape(0) != K) {
        throw ArgumentError(
            "Shape mismatch: A is [" + std::to_string(M) + "×" + std::to_string(K) +
            "] but B is [" + std::to_string(B.shape(0)) + "×" + std::to_string(N) + "]");
    }

    py::array_t<float> C_out;
    if (C.is_none()) {
        // Allocate output
        C_out = py::array_t<float>({M, N});
        float* ptr = C_out.mutable_data();
        if (beta != 0.0f) {
            // Zero-initialize when beta=0 (default) is implied but no C given
            for (int i = 0; i < M * N; ++i) ptr[i] = 0.0f;
        }
    } else {
        C_out = output_matrix(C, M, N);
    }

    const float* A_ptr = static_cast<const float*>(A.data());
    const float* B_ptr = static_cast<const float*>(B.data());
    float*       C_ptr = C_out.mutable_data();

    {
        py::gil_scoped_release release;
        IsaOverrideGuard guard(isa);   // no-op restore if isa == "" (auto)
        simd_ml::gemm::sgemm_packed(M, N, K, alpha, A_ptr, K, B_ptr, N, beta, C_ptr, N);
    }
    return C_out;
}

// ─── Binding: gelu / gelu_inplace ────────────────────────────────────────────
// approximate="none" (default, as in PyTorch): exact 0.5·x·(1 + erf(x/√2)).
// approximate="tanh": the tanh formula, PyTorch's approximate='tanh'.
using GeluFn = void (*)(const float*, float*, std::size_t);

static GeluFn gelu_variant(const std::string& approximate) {
    if (approximate == "none") return &gelu_erf_forward_avx2;
    if (approximate == "tanh") return &gelu_forward_avx2;
    throw ArgumentError("approximate must be 'none' or 'tanh' (got '" + approximate + "')");
}

static void py_gelu_inplace(py::array x, const std::string& approximate = "none") {
    const GeluFn fn = gelu_variant(approximate);
    validate_array(x, "x");
    check_writeable(x, "x");

    std::size_t n = static_cast<std::size_t>(x.size());
    float* ptr = static_cast<float*>(x.mutable_data());

    {
        py::gil_scoped_release release;
        fn(ptr, ptr, n);
    }
}

static py::array_t<float> py_gelu(py::array x, const std::string& approximate = "none") {
    const GeluFn fn = gelu_variant(approximate);
    validate_array(x, "x");

    std::size_t n = static_cast<std::size_t>(x.size());
    py::array_t<float> out(x.request().shape);

    {
        py::gil_scoped_release release;
        fn(static_cast<const float*>(x.data()), out.mutable_data(), n);
    }
    return out;
}

// ─── Binding: relu ───────────────────────────────────────────────────────────
static py::array_t<float> py_relu(py::array x) {
    validate_array(x, "x");
    std::size_t n = static_cast<std::size_t>(x.size());
    py::array_t<float> out(x.request().shape);
    {
        py::gil_scoped_release release;
        activations::relu_avx2(static_cast<const float*>(x.data()),
                                out.mutable_data(), static_cast<int>(n));
    }
    return out;
}

// ─── Binding: silu ───────────────────────────────────────────────────────────
static py::array_t<float> py_silu(py::array x) {
    validate_array(x, "x");
    std::size_t n = static_cast<std::size_t>(x.size());
    py::array_t<float> out(x.request().shape);
    {
        py::gil_scoped_release release;
        activations::silu_avx2(static_cast<const float*>(x.data()),
                                out.mutable_data(), static_cast<int>(n));
    }
    return out;
}

// ─── Binding: softmax ────────────────────────────────────────────────────────
static py::array_t<float> py_softmax(py::array x, int axis = -1) {
    validate_array(x, "x");
    if (x.ndim() < 1) throw ArgumentError("softmax requires an array with at least 1 dimension");

    int ndim = static_cast<int>(x.ndim());
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw ArgumentError("softmax axis " + std::to_string(axis) + " out of range");
    if (axis != ndim - 1)
        throw ArgumentError(
            "softmax currently supports the last axis only (axis=-1). "
            "For other axes, transpose first.");

    auto buf = x.request();
    py::array_t<float> out(buf.shape);
    float* out_ptr      = out.mutable_data();
    const float* in_ptr = static_cast<const float*>(x.data());
    int row_size        = static_cast<int>(buf.shape[ndim - 1]);
    int rows            = 1;
    for (int i = 0; i < ndim - 1; ++i)
        rows = static_cast<int>(rows * buf.shape[i]);

    {
        py::gil_scoped_release release;
        for (int r = 0; r < rows; ++r) {
            const float* row_in  = in_ptr  + static_cast<std::size_t>(r) * row_size;
            float*       row_out = out_ptr + static_cast<std::size_t>(r) * row_size;
            activations::softmax_row_avx2(row_in, row_out, row_size);
        }
    }
    return out;
}

// ─── Binding: layer_norm ─────────────────────────────────────────────────────
/**
 * layer_norm(x, gamma=None, beta=None, eps=1e-5) -> np.ndarray
 *
 * Applies Layer Normalization over the last axis.
 * output = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * gamma and beta must have the same shape as x's last dimension.
 */
static py::array_t<float> py_layer_norm(py::array x,
                                          py::object gamma_obj = py::none(),
                                          py::object beta_obj  = py::none(),
                                          float eps = 1e-5f) {
    validate_array(x, "x");
    if (x.ndim() < 1) throw ArgumentError("layer_norm requires an array with at least 1 dimension");

    int ndim     = static_cast<int>(x.ndim());
    int row_size = static_cast<int>(x.shape(ndim - 1));
    // An empty last axis has nothing to normalise; computing the row count
    // as size / row_size would divide by zero (crashed the interpreter).
    int rows     = row_size > 0 ? static_cast<int>(x.size()) / row_size : 0;

    // Validate gamma/beta if provided
    const float* gamma_ptr = nullptr;
    const float* beta_ptr  = nullptr;
    py::array_t<float> gamma_arr, beta_arr;

    if (!gamma_obj.is_none()) {
        gamma_arr = gamma_obj.cast<py::array_t<float>>();
        validate_array(gamma_arr, "gamma");
        if (gamma_arr.size() != row_size)
            throw ArgumentError("gamma must have size equal to the last dimension of x");
        gamma_ptr = static_cast<const float*>(gamma_arr.data());
    }
    if (!beta_obj.is_none()) {
        beta_arr = beta_obj.cast<py::array_t<float>>();
        validate_array(beta_arr, "beta");
        if (beta_arr.size() != row_size)
            throw ArgumentError("beta must have size equal to the last dimension of x");
        beta_ptr = static_cast<const float*>(beta_arr.data());
    }

    auto buf = x.request();
    py::array_t<float> out(buf.shape);
    const float* in_ptr = static_cast<const float*>(x.data());
    float*       out_ptr = out.mutable_data();

    {
        py::gil_scoped_release release;
        for (int r = 0; r < rows; ++r) {
            const float* row_in  = in_ptr  + static_cast<std::size_t>(r) * row_size;
            float*       row_out = out_ptr + static_cast<std::size_t>(r) * row_size;
            activations::layer_norm_avx2(row_in, row_out, row_size, gamma_ptr, beta_ptr, eps);
        }
    }
    return out;
}

// ─── Binding: set_num_threads / get_num_threads ──────────────────────────────
#ifdef SIMD_ML_OPENMP
static constexpr bool kHasOpenMP = true;
#else
static constexpr bool kHasOpenMP = false;
#endif

static void py_set_num_threads(int n) {
    if (n <= 0) throw std::invalid_argument("num_threads must be positive");
    // Without OpenMP the GEMM is single-threaded and this call cannot take
    // effect; it used to be silently ignored (get_num_threads() stayed 1).
    if (!kHasOpenMP && n > 1) {
        if (PyErr_WarnEx(PyExc_RuntimeWarning,
                         "set_num_threads(n > 1) has no effect: simd_ml_kernels was built "
                         "without OpenMP (CMake option SIMD_ML_OPENMP=ON enables it). "
                         "GEMM runs on 1 thread; see build_info()['openmp'].", 1) != 0)
            throw py::error_already_set();  // warnings configured as errors
    }
    simd_ml::gemm::set_num_threads(n);
}

static int py_get_num_threads() {
    return simd_ml::gemm::get_num_threads();
}

// ─── Binding: build_info ─────────────────────────────────────────────────────
static py::dict build_info() {
    py::dict info;
#ifdef __AVX512F__
    info["isa_compiled"] = "AVX-512F+DQ";
#elif defined(__AVX2__)
    info["isa_compiled"] = "AVX2+FMA3";
#else
    info["isa_compiled"] = "scalar (no AVX2)";
#endif
#ifdef __FMA__
    info["fma"] = true;
#else
    info["fma"] = false;
#endif
    info["alignment"]       = "64-byte (C++17 aligned operator new)";
    info["build_timestamp"] = std::string(__DATE__) + " " + std::string(__TIME__);
    info["runtime_isa"]     = simd_ml::dispatch::detected_isa();
    info["openmp"]          = kHasOpenMP;
    info["version"]         = SIMD_ML_VERSION;
    info["git_commit"]      = SIMD_ML_GIT_COMMIT;
    info["compiler"]        = SIMD_ML_COMPILER;
    info["kernel_flags"]    = SIMD_ML_KERNEL_FLAGS;
    return info;
}

// ─── Binding: is_aligned ─────────────────────────────────────────────────────
static bool py_is_aligned(py::array arr) {
    return is_aligned(arr.data());
}

// ─── Module definition ────────────────────────────────────────────────────────

// ─── Binding: sgemm_f16 ──────────────────────────────────────────────────────
/**
 * sgemm_f16(A, B, alpha=1.0, beta=0.0, C=None) -> np.ndarray[float32]
 *
 * Half-precision GEMM: C[fp32] = alpha * A[fp16] @ B[fp16] + beta * C[fp32]
 *
 * A and B must be numpy arrays with dtype=np.float16. C (output) is float32
 * to avoid overflow in intermediate sums. Raises RuntimeError if F16C is not
 * available on this CPU (check simd_ml_kernels.f16c_available() first).
 */
static py::array_t<float> py_sgemm_f16(py::array A,
                                         py::array B,
                                         float alpha   = 1.0f,
                                         float beta    = 0.0f,
                                         py::object C  = py::none()) {
    // Check F16C availability (runtime, not compile-time)
    if (!simd_ml::gemm::f16c_available()) {
        throw std::runtime_error(
            "sgemm_f16 requires F16C instructions which are not available on "
            "this CPU. Check simd_ml_kernels.f16c_available() before calling.");
    }

    // Validate A and B: must be float16 (kind='f', itemsize=2) and C-contiguous
    auto check_f16 = [](const py::array& arr, const char* name) {
        if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != 2) {
            throw DTypeError(
                std::string(name) + " must be float16 (dtype=np.float16). "
                "Call arr.astype(np.float16) before passing.");
        }
        if (!arr.attr("flags").attr("c_contiguous").cast<bool>()) {
            throw ArgumentError(
                std::string(name) + " must be C-contiguous. "
                "Call numpy.ascontiguousarray(arr) before passing.");
        }
    };
    check_f16(A, "A");
    check_f16(B, "B");
    if (A.ndim() != 2) throw ArgumentError("A must be 2-D");
    if (B.ndim() != 2) throw ArgumentError("B must be 2-D");

    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));

    if (B.shape(0) != static_cast<py::ssize_t>(K)) {
        throw ArgumentError(
            "Shape mismatch: A is [" + std::to_string(M) + "×" +
            std::to_string(K) + "] but B is [" +
            std::to_string(B.shape(0)) + "×" + std::to_string(N) + "]");
    }

    // Allocate or validate float32 C output
    py::array_t<float> C_out;
    if (C.is_none()) {
        C_out = py::array_t<float>({M, N});
        float* ptr = C_out.mutable_data();
        if (beta != 0.0f) {
            for (int i = 0; i < M * N; ++i) ptr[i] = 0.0f;
        }
    } else {
        C_out = output_matrix(C, M, N);
    }

    const auto* A_ptr = static_cast<const uint16_t*>(A.data());
    const auto* B_ptr = static_cast<const uint16_t*>(B.data());
    float*      C_ptr = C_out.mutable_data();

    {
        py::gil_scoped_release release;
        simd_ml::gemm::sgemm_f16_avx2(M, N, K,
                                       alpha, A_ptr, K, B_ptr, N,
                                       beta,  C_ptr, N);
    }
    return C_out;
}

// ─── Binding: f16c_available ──────────────────────────────────────────────────
static bool py_f16c_available() {
    return simd_ml::gemm::f16c_available();
}

// ─── Binding: sgemm_bf16 ─────────────────────────────────────────────────────
/**
 * sgemm_bf16(A, B, alpha=1.0, beta=0.0, C=None) -> np.ndarray[float32]
 *
 * BF16-input GEMM: C[fp32] = alpha * A[bf16] @ B[bf16] + beta * C[fp32]
 * A and B are uint16 arrays of BF16 bit patterns or ml_dtypes.bfloat16 arrays
 * (NumPy has no bfloat16 dtype of its own). Raises DTypeError for any other
 * dtype — float16 included — and ArgumentError for shape/layout problems.
 */
static py::array_t<float> py_sgemm_bf16(py::array A,
                                          py::array B,
                                          float alpha  = 1.0f,
                                          float beta   = 0.0f,
                                          py::object C = py::none()) {
    // Accepted: uint16 arrays holding BF16 bit patterns, and ml_dtypes.bfloat16
    // arrays (NumPy itself has no bfloat16 dtype; ml_dtypes registers one with
    // kind 'V' and name "bfloat16"). float16 is rejected: it is IEEE binary16
    // (5-bit exponent, 10-bit mantissa), and reading its bits as BF16 (8-bit
    // exponent, 7-bit mantissa) gives wrong values without any error.
    auto check_bf16 = [](const py::array& arr, const char* name) {
        const auto dt = arr.dtype();
        const bool is_uint16 = dt.kind() == 'u' && dt.itemsize() == 2;
        const bool is_ml_dtypes_bf16 =
            dt.itemsize() == 2 && py::str(dt.attr("name")).cast<std::string>() == "bfloat16";
        if (is_uint16 || is_ml_dtypes_bf16) return;
        const std::string got = py::str(dt.attr("name"));
        std::string msg = std::string(name) +
            " must hold BF16 values: a uint16 array of BF16 bit patterns or an "
            "ml_dtypes.bfloat16 array (got " + got + ").";
        if (dt.kind() == 'f' && dt.itemsize() == 2)
            msg += " float16 is IEEE half precision, not BF16: use sgemm_f16 for float16 inputs.";
        throw DTypeError(msg);
    };
    check_bf16(A, "A");
    check_bf16(B, "B");
    if (A.ndim() != 2) throw ArgumentError("A must be 2-D");
    if (B.ndim() != 2) throw ArgumentError("B must be 2-D");
    check_contiguous(A, "A");
    check_contiguous(B, "B");

    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));

    if (B.shape(0) != static_cast<py::ssize_t>(K))
        throw ArgumentError(
            "Shape mismatch: A is [" + std::to_string(M) + "×" +
            std::to_string(K) + "] but B is [" +
            std::to_string(B.shape(0)) + "×" + std::to_string(N) + "]");

    // Allocate or validate FP32 C output
    py::array_t<float> C_out;
    if (C.is_none()) {
        C_out = py::array_t<float>({M, N});
        if (beta != 0.0f) {
            float* ptr = C_out.mutable_data();
            std::fill(ptr, ptr + M * N, 0.0f);
        }
    } else {
        C_out = output_matrix(C, M, N);
    }

    const auto* A_ptr = static_cast<const uint16_t*>(A.data());
    const auto* B_ptr = static_cast<const uint16_t*>(B.data());
    float*      C_ptr = C_out.mutable_data();

    {
        py::gil_scoped_release release;
        simd_ml::gemm::sgemm_bf16_avx2(M, N, K,
                                        alpha, A_ptr, K, B_ptr, N,
                                        beta,  C_ptr, N);
    }
    return C_out;
}

static bool py_bf16_avx512bf16_available() {
    return simd_ml::gemm::bf16_avx512bf16_available();
}

// ─── GEMMConfig C++ struct ───────────────────────────────────────────────────
/**
 * GEMMConfig — callable GEMM configuration object (roadmap §9 DX vision).
 *
 * Stores default alpha, beta, and isa values. Tile parameters (tile_m, tile_n,
 * tile_k, mr, nr) are accepted for API completeness and future auto-tuning
 * integration; they do not currently change dispatch (the compiled kernel uses
 * compile-time constants). See ROADMAP.md §v0.9.
 *
 * Python usage:
 *   gemm = sk.GEMMConfig(tile_m=128, tile_n=128, tile_k=256, mr=8, nr=8)
 *   C = gemm(A, B)
 *   C = sk.sgemm(A, B, isa="avx2")
 */
struct GEMMConfig {
    float       alpha  = 1.0f;
    float       beta   = 0.0f;
    std::string isa    = "";
    int         tile_m = 128;
    int         tile_n = 2048;
    int         tile_k = 256;
    int         mr     = 8;
    int         nr     = 8;

    // Validate isa string — called from Python __init__ via __post_init__ logic
    void validate() const {
        if (!isa.empty() && isa != "avx2" && isa != "avx512" && isa != "scalar") {
            throw py::value_error(
                "isa must be one of: 'avx2', 'avx512', 'scalar' (got '" + isa + "')");
        }
        for (auto [name, val] : std::initializer_list<std::pair<const char*, int>>{
                {"tile_m", tile_m}, {"tile_n", tile_n}, {"tile_k", tile_k},
                {"mr", mr}, {"nr", nr}}) {
            if (val <= 0)
                throw py::value_error(
                    std::string(name) + " must be positive, got " + std::to_string(val));
        }
    }

    // __call__: C = alpha * A @ B + beta * C
    py::array_t<float> call(
        py::array A, py::array B,
        py::object C_obj   = py::none(),
        py::object alpha_o = py::none(),
        py::object beta_o  = py::none()) const
    {
        float a = alpha_o.is_none() ? alpha : alpha_o.cast<float>();
        float b = beta_o.is_none()  ? beta  : beta_o.cast<float>();
        return py_sgemm(A, B, C_obj, a, b, isa);
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "GEMMConfig(alpha=" << alpha << ", beta=" << beta;
        if (!isa.empty()) ss << ", isa='" << isa << "'";
        if (tile_m != 128)  ss << ", tile_m=" << tile_m;
        if (tile_n != 2048) ss << ", tile_n=" << tile_n;
        if (tile_k != 256)  ss << ", tile_k=" << tile_k;
        if (mr != 8) ss << ", mr=" << mr;
        if (nr != 8) ss << ", nr=" << nr;
        ss << ")";
        return ss.str();
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "simd-ml-kernels: Hand-vectorized AVX2/AVX-512 microkernels for ML workloads.\n\n"
        "All operations are zero-copy: NumPy float32 array buffers are used directly.\n"
        "The GIL is released during computation for threading compatibility.\n\n"
        "Quick start:\n"
        "    import numpy as np, simd_ml_kernels as sk\n"
        "    A = np.random.randn(256, 256).astype(np.float32)\n"
        "    B = np.random.randn(256, 256).astype(np.float32)\n"
        "    C = sk.sgemm(A, B)           # SIMD GEMM\n"
        "    y = sk.gelu(x)               # Vectorized GeLU\n"
        "    print(sk.build_info())       # ISA and build details\n";

    // PyErr_NewException accepts a tuple of bases, giving each error two
    // builtin bases (see the struct definitions above).
    auto dtype_error = py::register_exception<DTypeError>(
        m, "DTypeError", py::make_tuple(py::handle(PyExc_TypeError), py::handle(PyExc_RuntimeError)));
    dtype_error.attr("__doc__") =
        "An array argument has the wrong dtype. Subclass of TypeError; also of "
        "RuntimeError (raised before 0.3) until 0.5.0.";
    auto argument_error = py::register_exception<ArgumentError>(
        m, "ArgumentError", py::make_tuple(py::handle(PyExc_ValueError), py::handle(PyExc_RuntimeError)));
    argument_error.attr("__doc__") =
        "An argument has the wrong shape, layout (C-contiguity, writeability) or "
        "value. Subclass of ValueError; also of RuntimeError (raised before 0.3) "
        "until 0.5.0.";

    m.def("sgemm", &py_sgemm,
          py::arg("A"), py::arg("B"), py::arg("C") = py::none(),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("isa") = "",
          R"doc(
SIMD GEMM: computes C = alpha * A @ B + beta * C  (float32).

Parameters
----------
A     : np.ndarray[float32], shape [M, K], C-contiguous
B     : np.ndarray[float32], shape [K, N], C-contiguous
C     : np.ndarray[float32], shape [M, N], optional (allocated if None)
alpha : float, default 1.0
beta  : float, default 0.0  (0 = overwrite C, 1 = accumulate into C)
isa   : str, default "" (auto-detect). One of "", "avx2", "avx512".
        Forces this specific call to use the named kernel instead of the
        runtime auto-detected one — e.g. for benchmarking AVX2 vs AVX-512
        on the same AVX-512-capable machine. Requesting "avx512" on
        hardware without AVX-512F+DQ raises RuntimeError (check
        simd_ml_kernels.avx512_available() first). This override applies only
        to this call — it does not change behavior for subsequent calls.
        "scalar" is validated as a recognized value but raises
        RuntimeError ("not yet implemented") rather than silently falling
        back, since no forced full-matrix scalar path exists yet.

Returns
-------
np.ndarray[float32] of shape [M, N] — same object as C if provided.

Notes
-----
Uses a Goto/BLIS-style packed GEMM with register blocking (8×8 AVX2 or
8×16 AVX-512 micro-tile, auto-dispatched at runtime unless isa= forces a
choice), panel packing, and FMA intrinsics. Achieves 55–59% of OpenBLAS
single-threaded throughput for N≥256 (measured; see docs/BENCHMARKS.md). Small matrices (N≤64) lag further
behind (~4%) and can even be slower than a plain scalar loop — packing
overhead dominates and is not yet amortised at that size, and wider SIMD
doesn't help a packing-bound regime — see DESIGN.md §7 for the full gap
analysis.
)doc");

    static const char* gelu_doc = R"doc(
GeLU activation, element-wise, float32. Same definitions as
torch.nn.functional.gelu:

approximate="none" (default): exact  0.5*x*(1 + erf(x/sqrt(2)))
approximate="tanh":           0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

Accuracy against a float64 evaluation of the same formula (asserted in
tests/test_precision.py::TestGeLU, dense sweep of [-10, 10]):
  "none": max abs error <= 2e-6 (measured 9.5e-7); <= 1e-6 on [-3, 3]
          (measured 4.0e-7). erf is a rational approximation (XLA
          ErfImpl32 coefficients). For x < -4, where |GeLU(x)| < 1.3e-4,
          the absolute bound holds but the relative error is large.
  "tanh": max abs error <= 2e-6 vs the tanh formula (measured 1.5e-6). It
          differs from the exact GeLU by up to 4.7e-4 (formula difference).
Both: x < -10 returns -0 (|GeLU(-10)| = 7.6e-23); GeLU(+inf) = +inf,
GeLU(-inf) = -0, GeLU(NaN) = NaN.

Parameters
----------
x           : np.ndarray[float32], any shape, C-contiguous
approximate : "none" | "tanh"
)doc";

    // pybind11 keeps the doc pointer, so the strings need static storage.
    static const std::string gelu_out_doc =
        std::string(gelu_doc) + "\nReturns a new array; x is not modified.\n";
    static const std::string gelu_inplace_doc =
        std::string(gelu_doc) + "\nIn place: x[:] = gelu(x); x must be writeable.\n";
    m.def("gelu", &py_gelu, py::arg("x"), py::arg("approximate") = "none",
          gelu_out_doc.c_str());
    m.def("gelu_inplace", &py_gelu_inplace, py::arg("x"), py::arg("approximate") = "none",
          gelu_inplace_doc.c_str());

    m.def("relu", &py_relu,
          py::arg("x"),
          "Return ReLU(x) = max(0, x) computed with AVX2 vectorization.");

    m.def("silu", &py_silu,
          py::arg("x"),
          "Return SiLU(x) = x * sigmoid(x) using AVX2 vectorization.");

    m.def("softmax", &py_softmax,
          py::arg("x"), py::arg("axis") = -1,
          "Return numerically stable softmax over the last axis (or specified axis).");

    m.def("layer_norm", &py_layer_norm,
          py::arg("x"),
          py::arg("gamma") = py::none(),
          py::arg("beta")  = py::none(),
          py::arg("eps")   = 1e-5f,
          R"doc(
Apply Layer Normalization over the last axis using AVX2 vectorization.

output = (x - mean) / sqrt(var + eps) * gamma + beta

Parameters
----------
x     : np.ndarray[float32], any shape
gamma : np.ndarray[float32], shape [last_dim], optional scale
beta  : np.ndarray[float32], shape [last_dim], optional shift
eps   : float, numerical stability constant (default 1e-5)

Returns
-------
np.ndarray[float32] same shape as x.
)doc");

    m.def("set_num_threads", &py_set_num_threads,
          py::arg("n"),
          "Set the number of OpenMP threads used by GEMM. In a build without OpenMP\n"
          "(build_info()['openmp'] is False) GEMM is single-threaded and n > 1 emits\n"
          "a RuntimeWarning instead of being silently ignored.");

    m.def("get_num_threads", &py_get_num_threads,
          "Return the current OpenMP thread count used by GEMM.");

    m.def("build_info", &build_info,
          "Return a dict with ISA, FMA support, alignment, and build details.");

    m.def("detected_isa", []() {
        return std::string(simd_ml::dispatch::detected_isa());
    }, "Return the runtime-detected ISA label (avx512|avx2|sse42|scalar).");

    m.def("avx512_available", []() {
        return simd_ml::gemm::gemm_packed_avx512_hardware_available();
    }, R"doc(
Return True if this CPU supports AVX-512F+DQ (the instructions required by
sgemm_packed's 8x16 micro-kernel), independent of any isa= override.

Useful for checking before requesting isa='avx512' explicitly:

>>> if simd_ml_kernels.avx512_available():
...     C = simd_ml_kernels.sgemm(A, B, isa='avx512')
)doc");

    m.def("bf16_avx512bf16_available", &py_bf16_avx512bf16_available,
          "Return True if this CPU supports AVX-512 BF16 (vdpbf16ps). "
          "Not required for sgemm_bf16 (which uses AVX2); indicates "
          "availability of the higher-throughput packing path (v1.0 roadmap).");

    m.def("sgemm_bf16", &py_sgemm_bf16,
          py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("C") = py::none(),
          R"doc(
BF16-input GEMM: C[float32] = alpha * A[bfloat16] @ B[bfloat16] + beta * C[float32]

A and B store Brain Float 16 values (same exponent as FP32, 7-bit mantissa).
Output C is always float32 — BF16 accumulation would lose mantissa bits at
each step. Requires only AVX2 (Haswell+); no special -mavx512bf16 flag.

BF16→FP32 conversion is exact: BF16 is the top 16 bits of an FP32 value, so
zero-extending to 32 bits and shifting left by 16 (vpmovzxwd + vpslld, AVX2)
reconstructs it.

Input formats accepted
----------------------
uint16:    BF16 bit patterns. From float32 (truncating):
           (x.view(np.uint32) >> 16).astype(np.uint16)
ml_dtypes: arrays with dtype=ml_dtypes.bfloat16.
PyTorch:   bf16_tensor.view(torch.uint16).numpy()
float16 is rejected with DTypeError: it is IEEE half precision, a different
format (use sgemm_f16). NumPy itself has no bfloat16 dtype.

Parameters
----------
A     : ndarray [M, K], uint16 BF16 bit patterns or ml_dtypes.bfloat16, C-contiguous
B     : ndarray [K, N], uint16 BF16 bit patterns or ml_dtypes.bfloat16, C-contiguous
alpha : float, default 1.0
beta  : float, default 0.0  (0 = overwrite C, 1 = accumulate into C)
C     : ndarray[float32], shape [M, N], optional (allocated if None)

Returns
-------
ndarray[float32], shape [M, N]

Example
-------
>>> import numpy as np, simd_ml_kernels as sk
>>> to_bf16 = lambda x: (x.view(np.uint32) >> 16).astype(np.uint16)
>>> A = np.random.randn(64, 256).astype(np.float32)
>>> B = np.random.randn(256, 32).astype(np.float32)
>>> C = sk.sgemm_bf16(to_bf16(A), to_bf16(B))   # float32 [64, 32]
)doc");

    m.def("f16c_available", &py_f16c_available,
          "Return True if this CPU supports F16C instructions (required by sgemm_f16).");

    m.def("sgemm_f16", &py_sgemm_f16,
          py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("C") = py::none(),
          R"doc(
FP16-input GEMM: C[float32] = alpha * A[float16] @ B[float16] + beta * C[float32]

A and B must have dtype=np.float16 (IEEE 754 half-precision, 16-bit).
C (output) is always float32 — accumulation in FP16 would lose precision
at K > 64, so the output stays in full precision.

Requires F16C instructions (Haswell+, AMD Piledriver+). Check
simd_ml_kernels.f16c_available() before calling; raises RuntimeError if absent.

Parameters
----------
A     : ndarray[float16], shape [M, K], C-contiguous
B     : ndarray[float16], shape [K, N], C-contiguous
alpha : float, default 1.0
beta  : float, default 0.0  (0 = overwrite C, 1 = accumulate into C)
C     : ndarray[float32], shape [M, N], optional (allocated if None)

Returns
-------
ndarray[float32] of shape [M, N]

Notes
-----
Uses F16C (_mm256_cvtph_ps) to convert 8 FP16 values to FP32 in a single
instruction, then accumulates with AVX2 FMA (_mm256_fmadd_ps). No panel
packing — appropriate for the batch sizes (1–128) common in LLM inference
where the entire working set fits in L2 cache.

Example
-------
>>> import numpy as np, simd_ml_kernels as sk
>>> if sk.f16c_available():
...     A = np.random.randn(64, 256).astype(np.float16)
...     B = np.random.randn(256, 64).astype(np.float16)
...     C = sk.sgemm_f16(A, B)   # float32 output
)doc");

    m.def("is_aligned", &py_is_aligned,
          py::arg("arr"),
          "Return True if the array's data buffer is 64-byte aligned.");


    // ─── GEMMConfig ──────────────────────────────────────────────────────────────
    py::class_<GEMMConfig>(m, "GEMMConfig",
        R"doc(
Callable GEMM configuration object (roadmap §9 DX vision).

Stores default alpha, beta, and isa values for repeated GEMM calls with
consistent configuration. Tile parameters are accepted for API completeness
and future auto-tuning integration.

Parameters
----------
alpha   : float, default 1.0    — scalar multiplier for A@B
beta    : float, default 0.0    — scalar multiplier for C (0 = overwrite)
isa     : str, default ""       — ISA hint: "", "avx2", "avx512", "scalar"
tile_m  : int, default 128      — M-panel tile (future auto-tuning)
tile_n  : int, default 2048     — N-panel tile (future auto-tuning)
tile_k  : int, default 256      — K-panel tile (future auto-tuning)
mr      : int, default 8        — micro-kernel row block (future)
nr      : int, default 8        — micro-kernel col block (future)

Examples
--------
>>> import numpy as np, simd_ml_kernels as sk
>>> A = np.random.randn(256, 512).astype(np.float32)
>>> B = np.random.randn(512, 128).astype(np.float32)
>>>
>>> gemm = sk.GEMMConfig()
>>> C = gemm(A, B)
>>>
>>> gemm = sk.GEMMConfig(alpha=2.0, beta=0.5, isa="avx2")
>>> C_init = np.zeros((256, 128), dtype=np.float32)
>>> C = gemm(A, B, C=C_init)    # C = 2*A@B + 0.5*C_init
>>>
>>> gemm = sk.GEMMConfig(tile_m=128, tile_n=128, tile_k=256, mr=8, nr=8)
>>> C = gemm(A, B)
)doc")
        .def(py::init([](float alpha, float beta, std::string isa,
                         int tile_m, int tile_n, int tile_k, int mr, int nr) {
                GEMMConfig cfg;
                cfg.alpha = alpha; cfg.beta = beta; cfg.isa = std::move(isa);
                cfg.tile_m = tile_m; cfg.tile_n = tile_n; cfg.tile_k = tile_k;
                cfg.mr = mr; cfg.nr = nr;
                cfg.validate();
                return cfg;
             }),
             py::arg("alpha")  = 1.0f,
             py::arg("beta")   = 0.0f,
             py::arg("isa")    = "",
             py::arg("tile_m") = 128,
             py::arg("tile_n") = 2048,
             py::arg("tile_k") = 256,
             py::arg("mr")     = 8,
             py::arg("nr")     = 8)
        .def("__call__", &GEMMConfig::call,
             py::arg("A"), py::arg("B"),
             py::arg("C")     = py::none(),
             py::arg("alpha") = py::none(),
             py::arg("beta")  = py::none(),
             R"doc(
Compute C = alpha * A @ B + beta * C using stored configuration.

Parameters
----------
A     : ndarray[float32], shape [M, K]
B     : ndarray[float32], shape [K, N]
C     : ndarray[float32], shape [M, N], optional (allocated if None)
alpha : float, optional — overrides stored alpha for this call only
beta  : float, optional — overrides stored beta  for this call only
)doc")
        .def("__repr__", &GEMMConfig::repr)
        .def_readwrite("alpha",  &GEMMConfig::alpha,
                       "Scalar multiplier for A@B.")
        .def_readwrite("beta",   &GEMMConfig::beta,
                       "Scalar multiplier for existing C (0 = overwrite).")
        .def_readwrite("isa",    &GEMMConfig::isa,
                       "ISA hint string: '', 'avx2', 'avx512', or 'scalar'.")
        .def_readwrite("tile_m", &GEMMConfig::tile_m, "M-panel tile size.")
        .def_readwrite("tile_n", &GEMMConfig::tile_n, "N-panel tile size.")
        .def_readwrite("tile_k", &GEMMConfig::tile_k, "K-panel tile size.")
        .def_readwrite("mr",     &GEMMConfig::mr,     "Micro-kernel row block.")
        .def_readwrite("nr",     &GEMMConfig::nr,     "Micro-kernel col block.");

}
