/* Run only on a CPU without AVX: advanced entry points must not be called. */
#include "../src/similarity.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
static _Atomic unsigned long advanced_calls;
int __wrap_similarity_avx_cosine(const unsigned char *a, const unsigned char *b, size_t x, size_t y,
                                 double *z) {
    (void)a;
    (void)b;
    (void)x;
    (void)y;
    (void)z;
    atomic_fetch_add(&advanced_calls, 1);
    return 0;
}
int __wrap_similarity_avx2_cosine(const unsigned char *a, const unsigned char *b, size_t x,
                                  size_t y, double *z) {
    (void)a;
    (void)b;
    (void)x;
    (void)y;
    (void)z;
    atomic_fetch_add(&advanced_calls, 1);
    return 0;
}
static void *worker(void *unused) {
    (void)unused;
    const unsigned char a[] = {0, 0, 128, 63, 0, 0, 0, 64};
    for (int i = 0; i < 10000; i++) {
        double result = -1;
        if (!similarity_cosine_f32le(a, a, 8, 8, &result) || fabs(result - 1.0) > 1e-12)
            return (void *)(uintptr_t)1;
        result = 17;
        if (similarity_cosine_f32le(a, a, 8, 4, &result) || result != 17)
            return (void *)(uintptr_t)1;
    }
    return NULL;
}
int main(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx")) {
        puts("requires a non-AVX CPU");
        return 2;
    }
#else
    return 2;
#endif
    pthread_t threads[8];
    for (int i = 0; i < 8; i++)
        if (pthread_create(&threads[i], NULL, worker, NULL))
            return 1;
    for (int i = 0; i < 8; i++) {
        void *r;
        if (pthread_join(threads[i], &r) || r)
            return 1;
    }
    if (atomic_load(&advanced_calls))
        return 1;
    puts("concurrent dispatch: no AVX/AVX2 calls");
    return 0;
}
