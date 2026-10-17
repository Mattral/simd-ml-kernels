/**
 * bench_stat.cpp — Statistical Benchmarking Harness with JSON Output
 *
 * This is the rigorous companion to main_bench.cpp. Where main_bench.cpp
 * uses RDTSC cycle counts, this harness uses high-resolution wall-clock timing
 * with statistical analysis:
 *
 *   - Configurable warmup and measurement repetitions
 *   - Reports: min, max, mean, median, std-dev, 95% CI (via t-distribution)
 *   - JSON output for longitudinal tracking and CI regression detection
 *   - CPU pinning via sched_setaffinity on Linux
 *   - Optional OpenBLAS comparison when built with -DBENCH_OPENBLAS=ON
 *
 * Usage:
 *   ./bench_stat [--sizes 64,128,256,512] [--reps 30] [--output results.json]
 *
 * Example (with freq lock for reproducible results):
 *   sudo cpupower frequency-set -g performance
 *   taskset -c 0 ./bench_stat --sizes 64,128,256,512,1024 --reps 50 \
 *     --output benchmarks/results/gemm_results.json
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <cassert>

#ifdef __linux__
#  include <sched.h>
#endif

#include "kernels/cache_alloc.hpp"
#include "kernels/gemm/avx2_gemm_packed.hpp"
#include "kernels/intrinsic_gelu.hpp"

#ifdef BENCH_WITH_OPENBLAS
extern "C" {
void sgemm_(char* transa, char* transb,
            int* m, int* n, int* k,
            float* alpha, float* A, int* lda,
            float* B, int* ldb,
            float* beta, float* C, int* ldc);
}
#endif

// --- Configuration ------------------------------------------------------------

struct BenchConfig {
    std::vector<int> gemm_sizes  = {64, 128, 256, 512, 1024};
    std::vector<int> gelu_sizes  = {1024, 16384, 262144, 1048576};
    int              warmup_reps = 5;
    int              meas_reps   = 30;
    std::string      output_path = "";
    bool             quiet       = false;
};

// --- Timer -------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

static inline double now_seconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// --- Statistics ---------------------------------------------------------------

struct Stats {
    double min_s;
    double max_s;
    double mean_s;
    double median_s;
    double stddev_s;
    double ci95_half;  // half-width of 95% confidence interval
};

static Stats compute_stats(std::vector<double>& times) {
    int n = static_cast<int>(times.size());
    assert(n > 0);

    std::sort(times.begin(), times.end());

    Stats s;
    s.min_s    = times.front();
    s.max_s    = times.back();
    s.median_s = (n % 2 == 0)
                    ? 0.5 * (times[n/2 - 1] + times[n/2])
                    : times[n/2];

    // Welford's online algorithm for mean and variance
    double m = 0.0, m2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double delta = times[i] - m;
        m  += delta / (i + 1);
        m2 += delta * (times[i] - m);
    }
    s.mean_s  = m;
    s.stddev_s = (n > 1) ? std::sqrt(m2 / (n - 1)) : 0.0;

    // 95% confidence interval using t-distribution approximation.
    // For n >= 30, t_{0.025, n-1} ≈ 1.96 + 0.5/sqrt(n) (very close to normal).
    // For smaller n, use a conservative upper bound of 2.5.
    double t95;
    if      (n >= 120) t95 = 1.980;
    else if (n >=  60) t95 = 2.000;
    else if (n >=  30) t95 = 2.042;
    else if (n >=  20) t95 = 2.093;
    else if (n >=  10) t95 = 2.262;
    else               t95 = 2.500;

    s.ci95_half = t95 * s.stddev_s / std::sqrt(static_cast<double>(n));
    return s;
}

// --- Benchmark helpers --------------------------------------------------------

static double gflops_sgemm(int N, double time_s) {
    // 2*N^3 FLOP for square GEMM (N multiply-adds)
    return 2.0 * static_cast<double>(N) * N * N / time_s * 1e-9;
}

static double gelems_per_sec(int n, double time_s) {
    return static_cast<double>(n) / time_s * 1e-9;
}

// --- JSON builder -------------------------------------------------------------

struct JsonDoc {
    std::ostringstream ss;
    bool first_entry = true;

    void begin() { ss << "{\n"; }

    void begin_array(const std::string& key) {
        if (!first_entry) ss << ",\n";
        ss << "  \"" << key << "\": [\n";
        first_entry = false;
    }

    void end_array() { ss << "\n  ]"; }

    void end() { ss << "\n}\n"; }

    void gemm_entry(const std::string& kernel, int N, const Stats& st,
                    bool first_in_arr) {
        if (!first_in_arr) ss << ",\n";
        ss << "    {\"kernel\":\"" << kernel << "\","
           << "\"N\":" << N << ","
           << "\"min_ms\":" << st.min_s * 1e3 << ","
           << "\"mean_ms\":" << st.mean_s * 1e3 << ","
           << "\"median_ms\":" << st.median_s * 1e3 << ","
           << "\"stddev_ms\":" << st.stddev_s * 1e3 << ","
           << "\"ci95_half_ms\":" << st.ci95_half * 1e3 << ","
           << "\"gflops_min_latency\":" << gflops_sgemm(N, st.min_s) << ","
           << "\"gflops_mean_latency\":" << gflops_sgemm(N, st.mean_s) << "}";
    }

    void gelu_entry(const std::string& kernel, int n, const Stats& st,
                    bool first_in_arr) {
        if (!first_in_arr) ss << ",\n";
        ss << "    {\"kernel\":\"" << kernel << "\","
           << "\"n\":" << n << ","
           << "\"min_ms\":" << st.min_s * 1e3 << ","
           << "\"mean_ms\":" << st.mean_s * 1e3 << ","
           << "\"median_ms\":" << st.median_s * 1e3 << ","
           << "\"stddev_ms\":" << st.stddev_s * 1e3 << ","
           << "\"ci95_half_ms\":" << st.ci95_half * 1e3 << ","
           << "\"gelems_per_sec\":" << gelems_per_sec(n, st.min_s) << "}";
    }

    std::string str() const { return ss.str(); }
};

// --- CPU affinity -------------------------------------------------------------

static void pin_to_core(int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        fprintf(stderr, "Warning: failed to pin to core %d (run as root or with CAP_SYS_NICE)\n",
                core_id);
    } else {
        fprintf(stderr, "Pinned to core %d\n", core_id);
    }
#else
    (void)core_id;
#endif
}

// --- Argument parsing ---------------------------------------------------------

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            cfg.gemm_sizes.clear();
            std::string tok;
            std::istringstream ss(argv[++i]);
            while (std::getline(ss, tok, ',')) {
                cfg.gemm_sizes.push_back(std::stoi(tok));
            }
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            cfg.meas_reps = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup_reps = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            cfg.quiet = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            printf("Usage: bench_stat [options]\n"
                   "  --sizes  N,M,...   GEMM square matrix sizes (default: 64,128,256,512,1024)\n"
                   "  --reps   N         Measurement repetitions (default: 30)\n"
                   "  --warmup N         Warmup repetitions (default: 5)\n"
                   "  --output FILE      Write JSON results to FILE\n"
                   "  --quiet            Suppress progress output\n");
            std::exit(0);
        } else {
            fprintf(stderr, "bench_stat: unknown argument '%s'\n"
                            "Run ./bench_stat --help for usage.\n", argv[i]);
            std::exit(1);
        }
    }
    return cfg;
}

// --- Main ---------------------------------------------------------------------

int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);
    pin_to_core(0);

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   simd-ml-kernels Statistical Benchmark (bench_stat)        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  Warmup reps: %d  |  Measurement reps: %d\n", cfg.warmup_reps, cfg.meas_reps);
    printf("  Reporting: min / mean ± 95%%CI / median  [GFLOPS]\n\n");

    JsonDoc doc;
    doc.begin();

    // --- GEMM Benchmarks -----------------------------------------------------
    printf("=== GEMM Benchmarks (C = A @ B, float32, single-threaded) ===\n");
    printf("  %-12s  %5s  %8s  %18s  %8s  %8s\n",
           "Kernel", "N", "min ms", "mean±CI ms", "median ms", "min GFLOPS");
    printf("  %-12s  %5s  %8s  %18s  %8s  %8s\n",
           "------", "---", "------", "----------", "---------", "-----------");

    doc.begin_array("gemm");
    bool first_gemm = true;

    for (int N : cfg.gemm_sizes) {
        auto A = make_aligned_array<float>(static_cast<std::size_t>(N) * N);
        auto B = make_aligned_array<float>(static_cast<std::size_t>(N) * N);
        auto C = make_aligned_array<float>(static_cast<std::size_t>(N) * N);

        // Fill with random-ish values
        for (int i = 0; i < N * N; ++i) {
            A[i] = (float)(i % 100) * 0.01f - 0.5f;
            B[i] = (float)((i * 17 + 3) % 100) * 0.01f - 0.5f;
        }

        // -- Packed SIMD GEMM ----------------------------------------------
        {
            std::vector<double> times(cfg.meas_reps);

            // Warmup
            for (int r = 0; r < cfg.warmup_reps; ++r) {
                std::fill(C.get(), C.get() + N * N, 0.0f);
                simd_ml::gemm::sgemm_packed(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C.get(), N);
            }

            // Measure
            for (int r = 0; r < cfg.meas_reps; ++r) {
                std::fill(C.get(), C.get() + N * N, 0.0f);
                double t0 = now_seconds();
                simd_ml::gemm::sgemm_packed(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C.get(), N);
                times[r] = now_seconds() - t0;
            }

            Stats s = compute_stats(times);
            printf("  %-12s  %5d  %8.3f  %8.3f±%6.3f  %8.3f  %8.2f\n",
                   "simd_packed", N,
                   s.min_s * 1e3, s.mean_s * 1e3, s.ci95_half * 1e3,
                   s.median_s * 1e3, gflops_sgemm(N, s.min_s));
            doc.gemm_entry("simd_packed", N, s, first_gemm);
            first_gemm = false;
        }

#ifdef BENCH_WITH_OPENBLAS
        // -- OpenBLAS comparison -------------------------------------------
        {
            std::vector<double> times(cfg.meas_reps);
            char transa = 'N', transb = 'N';
            float alpha = 1.0f, beta = 0.0f;
            int n = N;

            for (int r = 0; r < cfg.warmup_reps; ++r) {
                std::fill(C.get(), C.get() + N * N, 0.0f);
                sgemm_(&transa, &transb, &n, &n, &n, &alpha,
                        A.get(), &n, B.get(), &n, &beta, C.get(), &n);
            }
            for (int r = 0; r < cfg.meas_reps; ++r) {
                std::fill(C.get(), C.get() + N * N, 0.0f);
                double t0 = now_seconds();
                sgemm_(&transa, &transb, &n, &n, &n, &alpha,
                        A.get(), &n, B.get(), &n, &beta, C.get(), &n);
                times[r] = now_seconds() - t0;
            }
            Stats s = compute_stats(times);
            printf("  %-12s  %5d  %8.3f  %8.3f±%6.3f  %8.3f  %8.2f\n",
                   "openblas", N,
                   s.min_s * 1e3, s.mean_s * 1e3, s.ci95_half * 1e3,
                   s.median_s * 1e3, gflops_sgemm(N, s.min_s));
            doc.gemm_entry("openblas", N, s, first_gemm);
            first_gemm = false;
        }
#endif
    }
    doc.end_array();

    // --- GeLU Benchmarks -----------------------------------------------------
    printf("\n=== GeLU Benchmarks (element-wise, float32) ===\n");
    printf("  %-12s  %8s  %8s  %18s  %8s  %12s\n",
           "Kernel", "n", "min ms", "mean±CI ms", "median ms", "GElems/s");
    printf("  %-12s  %8s  %8s  %18s  %8s  %12s\n",
           "------", "-", "------", "----------", "---------", "--------");

    doc.begin_array("gelu");
    bool first_gelu = true;

    for (int n : cfg.gelu_sizes) {
        auto x_in  = make_aligned_array<float>(n);
        auto x_out = make_aligned_array<float>(n);
        for (int i = 0; i < n; ++i) x_in[i] = (float)(i % 1000) * 0.01f - 5.0f;

        // -- SIMD GeLU ----------------------------------------------------
        {
            std::vector<double> times(cfg.meas_reps);
            for (int r = 0; r < cfg.warmup_reps; ++r)
                gelu_forward_avx2(x_in.get(), x_out.get(), n);
            for (int r = 0; r < cfg.meas_reps; ++r) {
                double t0 = now_seconds();
                gelu_forward_avx2(x_in.get(), x_out.get(), n);
                times[r] = now_seconds() - t0;
            }
            Stats s = compute_stats(times);
            printf("  %-12s  %8d  %8.3f  %8.3f±%6.3f  %8.3f  %12.3f\n",
                   "gelu_avx2", n,
                   s.min_s * 1e3, s.mean_s * 1e3, s.ci95_half * 1e3,
                   s.median_s * 1e3, gelems_per_sec(n, s.min_s));
            doc.gelu_entry("gelu_avx2", n, s, first_gelu);
            first_gelu = false;
        }
    }
    doc.end_array();
    doc.end();

    // --- Output JSON ----------------------------------------------------------
    if (!cfg.output_path.empty()) {
        std::ofstream f(cfg.output_path);
        if (f.is_open()) {
            f << doc.str();
            printf("\nResults written to %s\n", cfg.output_path.c_str());
        } else {
            fprintf(stderr, "Warning: could not open %s for writing\n",
                    cfg.output_path.c_str());
        }
    }

    printf("\n");
    printf("Notes:\n");
    printf("  - min latency ≈ best-case (cache-warm) performance.\n");
    printf("  - mean±CI gives a 95%% confidence interval under realistic scheduling noise.\n");
    printf("  - For reproducible results: lock CPU frequency, pin to a quiet core.\n");
    printf("    sudo cpupower frequency-set -g performance\n");
    printf("    taskset -c 0 ./bench_stat --reps 50 --output results.json\n");

    return 0;
}
