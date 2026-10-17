#pragma once

/**
 * cache_alloc.hpp — 64-Byte Aligned Memory Allocation Layer
 *
 * Problem Context:
 *   Modern x86-64 CPUs fetch memory in 64-byte cache lines. An unaligned array
 *   that straddles a cache-line boundary forces TWO memory fetches for a single
 *   SIMD load, halving effective bandwidth and causing AVX store-forwarding stalls.
 *
 *   For GEMM kernels processing FP32 matrices:
 *     - AVX2 __m256 register = 256 bits = 8 × float32 = 32 bytes
 *     - AVX-512 __m512 register = 512 bits = 16 × float32 = 64 bytes (= 1 cache line)
 *     - Aligning to 64 bytes ensures every AVX-512 load hits exactly one cache line.
 *
 * Strategy:
 *   Use C++17 aligned operator new / delete (std::align_val_t) to guarantee
 *   alignment. It is standard on every supported toolchain (GCC, Clang,
 *   MinGW, MSVC), so no platform-specific allocator branches are needed.
 *   Provide an STL-compatible AlignedAllocator<T> so std::vector can use it,
 *   plus raw alloc/free helpers for flat C arrays used inside the GEMM kernel.
 */

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>

// ─── Alignment constant ──────────────────────────────────────────────────────
// 64 bytes == one cache line == one AVX-512 register width
static constexpr std::size_t CACHE_LINE_BYTES = 64;

// ─── Raw aligned allocation helpers ──────────────────────────────────────────

/**
 * aligned_alloc_raw — allocate `bytes` bytes aligned to `alignment`.
 * alignment must be a power-of-two and ≥ sizeof(void*).
 * Returns nullptr on allocation failure.
 */
inline void* aligned_alloc_raw(std::size_t bytes,
                                std::size_t alignment = CACHE_LINE_BYTES) noexcept {
    if (bytes == 0) return nullptr;
    return ::operator new(bytes, std::align_val_t(alignment), std::nothrow);
}

/**
 * aligned_free_raw — release memory allocated by aligned_alloc_raw.
 * `alignment` must equal the value passed to aligned_alloc_raw.
 */
inline void aligned_free_raw(void* ptr,
                             std::size_t alignment = CACHE_LINE_BYTES) noexcept {
    if (!ptr) return;
    ::operator delete(ptr, std::align_val_t(alignment));
}

/**
 * make_aligned_array<T> — allocate count elements of T, zero-initialised,
 * aligned to CACHE_LINE_BYTES. Returns a unique_ptr with a custom deleter.
 *
 * Usage:
 *   auto A = make_aligned_array<float>(M * K);  // cache-line aligned FP32 buffer
 */
template <typename T>
struct AlignedDeleter {
    std::size_t alignment = CACHE_LINE_BYTES;  // must match the allocation
    void operator()(T* ptr) const noexcept {
        aligned_free_raw(static_cast<void*>(ptr), alignment);
    }
};

template <typename T>
using AlignedUniquePtr = std::unique_ptr<T[], AlignedDeleter<T>>;

template <typename T>
AlignedUniquePtr<T> make_aligned_array(std::size_t count,
                                        std::size_t alignment = CACHE_LINE_BYTES) {
    void* raw = aligned_alloc_raw(count * sizeof(T), alignment);
    if (!raw) throw std::bad_alloc{};
    std::memset(raw, 0, count * sizeof(T));
    return AlignedUniquePtr<T>(static_cast<T*>(raw), AlignedDeleter<T>{alignment});
}

// ─── STL-Compatible AlignedAllocator<T> ──────────────────────────────────────
/**
 * Drop-in replacement for std::allocator that enforces 64-byte alignment.
 * Use with std::vector<float, AlignedAllocator<float>> for aligned heap buffers.
 *
 * Example:
 *   std::vector<float, AlignedAllocator<float>> matrix(M * N);
 *   // matrix.data() is guaranteed 64-byte aligned — safe for vmovaps / vloadd
 */
template <typename T, std::size_t Alignment = CACHE_LINE_BYTES>
class AlignedAllocator {
public:
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = aligned_alloc_raw(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc{};
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t /*n*/) noexcept {
        aligned_free_raw(static_cast<void*>(ptr), Alignment);
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

// ─── Alignment verification utility ──────────────────────────────────────────
/**
 * is_aligned — runtime check that a pointer meets a given alignment.
 * Use in debug builds / unit tests to assert allocator correctness.
 */
inline bool is_aligned(const void* ptr,
                        std::size_t alignment = CACHE_LINE_BYTES) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

// ─── Prefetch helpers ─────────────────────────────────────────────────────────
/**
 * prefetch_l1 / prefetch_l2 — issue software prefetch hints to pull a cache
 * line into L1/L2 before it is needed by the GEMM inner loop.
 * Maps to __builtin_prefetch on GCC/Clang, _mm_prefetch on MSVC.
 */
#if defined(__GNUC__) || defined(__clang__)
inline void prefetch_l1(const void* ptr) noexcept {
    __builtin_prefetch(ptr, 0, 3);   // read, high temporal locality (L1)
}
inline void prefetch_l2(const void* ptr) noexcept {
    __builtin_prefetch(ptr, 0, 2);   // read, moderate locality (L2)
}
#elif defined(_MSC_VER)
#include <intrin.h>
inline void prefetch_l1(const void* ptr) noexcept {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
}
inline void prefetch_l2(const void* ptr) noexcept {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T1);
}
#else
inline void prefetch_l1(const void*) noexcept {}
inline void prefetch_l2(const void*) noexcept {}
#endif
