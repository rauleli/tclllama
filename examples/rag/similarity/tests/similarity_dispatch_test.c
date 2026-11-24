/* Exercise both implementations and concurrent stateless dispatch. */
#include "../src/similarity.h"
#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>

int similarity_scalar_cosine(const unsigned char *, const unsigned char *, size_t, size_t,
                             double *);
int similarity_sse2_cosine(const unsigned char *, const unsigned char *, size_t, size_t, double *);
int __real_similarity_sse2_cosine(const unsigned char *, const unsigned char *, size_t, size_t,
                                  double *);
int __real_similarity_avx_cosine(const unsigned char *, const unsigned char *, size_t, size_t,
                                 double *);
int __real_similarity_avx2_cosine(const unsigned char *, const unsigned char *, size_t, size_t,
                                  double *);
static _Atomic unsigned long sse2_calls;
static _Atomic unsigned long avx_calls;
static _Atomic unsigned long avx2_calls;
int __wrap_similarity_sse2_cosine(const unsigned char *a, const unsigned char *b, size_t na,
                                  size_t nb, double *out) {
    atomic_fetch_add_explicit(&sse2_calls, 1, memory_order_relaxed);
    return __real_similarity_sse2_cosine(a, b, na, nb, out);
}
int __wrap_similarity_avx_cosine(const unsigned char *a, const unsigned char *b, size_t na,
                                 size_t nb, double *out) {
    atomic_fetch_add_explicit(&avx_calls, 1, memory_order_relaxed);
    return __real_similarity_avx_cosine(a, b, na, nb, out);
}
int __wrap_similarity_avx2_cosine(const unsigned char *a, const unsigned char *b, size_t na,
                                  size_t nb, double *out) {
    atomic_fetch_add_explicit(&avx2_calls, 1, memory_order_relaxed);
    return __real_similarity_avx2_cosine(a, b, na, nb, out);
}
static void *worker(void *unused) {
    (void)unused;
    const unsigned char a[] = {0, 0, 128, 63, 0, 0, 0, 64}, b[] = {0, 0, 0, 64, 0, 0, 128, 63};
    for (int i = 0; i < 10000; i++) {
        double x = 0, y = 0, z = 0;
        if (!similarity_scalar_cosine(a, b, 8, 8, &x) || !similarity_sse2_cosine(a, b, 8, 8, &y) ||
            !similarity_cosine_f32le(a, b, 8, 8, &z) || fabs(x - y) > 1e-12 || fabs(y - z) > 1e-12)
            return (void *)(uintptr_t)1;
        z = 19;
        if (similarity_cosine_f32le(a, b, 8, 4, &z) || z != 19)
            return (void *)(uintptr_t)1;
    }
    return NULL;
}
int main(void) {
    pthread_t threads[8];
    for (int i = 0; i < 8; i++)
        if (pthread_create(&threads[i], NULL, worker, NULL))
            return 1;
    for (int i = 0; i < 8; i++) {
        void *result;
        if (pthread_join(threads[i], &result) || result)
            return 1;
    }
    /* Each worker makes 80,000 direct SSE2 calls and 160,000 calls through
     * the dispatcher (one valid vector and one rejected length). */
    unsigned long expected_sse2_calls = 80000;
    unsigned long expected_avx_calls = 0;
    unsigned long expected_avx2_calls = 0;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        expected_avx2_calls = 160000;
    } else if (__builtin_cpu_supports("avx")) {
        expected_avx_calls = 160000;
    } else {
        expected_sse2_calls += 160000;
    }
#else
    expected_sse2_calls += 160000;
#endif
    unsigned long actual_sse2 = atomic_load_explicit(&sse2_calls, memory_order_relaxed);
    unsigned long actual_avx = atomic_load_explicit(&avx_calls, memory_order_relaxed);
    unsigned long actual_avx2 = atomic_load_explicit(&avx2_calls, memory_order_relaxed);
    if (actual_sse2 != expected_sse2_calls || actual_avx != expected_avx_calls ||
        actual_avx2 != expected_avx2_calls) {
        fprintf(stderr, "dispatch counts: SSE2 %lu/%lu, AVX %lu/%lu, AVX2 %lu/%lu\n",
                actual_sse2, expected_sse2_calls, actual_avx, expected_avx_calls, actual_avx2,
                expected_avx2_calls);
        return 1;
    }
    puts("scalar, SSE2, selected SIMD path and concurrent dispatch passed");
    return 0;
}
