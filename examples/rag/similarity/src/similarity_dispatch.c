/* Stateless dispatch: query each call, avoiding shared initialization state. */
#include "similarity.h"
#if (defined(TCLEMBEDDING_HAVE_SSE2) || defined(TCLEMBEDDING_HAVE_AVX) ||                          \
     defined(TCLEMBEDDING_HAVE_AVX2)) &&                                                           \
    defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#ifdef TCLEMBEDDING_HAVE_SSE2
static int similarity_runtime_sse2(void) {
    unsigned int a, b, c, d;
    /* The x86-64 ABI requires the OS to preserve the XMM register state. */
    return __get_cpuid(1, &a, &b, &c, &d) && (d & bit_SSE2) != 0;
}
#endif
#if defined(TCLEMBEDDING_HAVE_AVX) || defined(TCLEMBEDDING_HAVE_AVX2)
static int similarity_runtime_avx(void) {
    unsigned int a, b, c, d, lo, hi;
    if (!__get_cpuid(1, &a, &b, &c, &d) ||
        (c & (bit_AVX | bit_XSAVE | bit_OSXSAVE)) != (bit_AVX | bit_XSAVE | bit_OSXSAVE))
        return 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    (void)hi;
    return (lo & 6u) == 6u;
}
#endif
#ifdef TCLEMBEDDING_HAVE_AVX2
static int similarity_runtime_avx2(void) {
    unsigned int a, b, c, d;
    return similarity_runtime_avx() && __get_cpuid_count(7, 0, &a, &b, &c, &d) &&
           (b & bit_AVX2) != 0;
}
#endif
#endif
int similarity_scalar_cosine(const unsigned char *, const unsigned char *, size_t, size_t,
                             double *);
int similarity_scalar_dot(const unsigned char *, const unsigned char *, size_t, size_t, double *);
#ifdef TCLEMBEDDING_HAVE_SSE2
int similarity_sse2_cosine(const unsigned char *, const unsigned char *, size_t, size_t, double *);
#endif
#ifdef TCLEMBEDDING_HAVE_AVX
int similarity_avx_cosine(const unsigned char *, const unsigned char *, size_t, size_t, double *);
#endif
#ifdef TCLEMBEDDING_HAVE_AVX2
int similarity_avx2_cosine(const unsigned char *, const unsigned char *, size_t, size_t, double *);
#endif

int similarity_cosine_f32le(const unsigned char *a, const unsigned char *b, size_t na, size_t nb,
                            double *out) {
#if defined(TCLEMBEDDING_HAVE_AVX2) && defined(__x86_64__) &&                                      \
    (defined(__GNUC__) || defined(__clang__))
    if (similarity_runtime_avx2())
        return similarity_avx2_cosine(a, b, na, nb, out);
#endif
#if defined(TCLEMBEDDING_HAVE_AVX) && defined(__x86_64__) &&                                       \
    (defined(__GNUC__) || defined(__clang__))
    if (similarity_runtime_avx())
        return similarity_avx_cosine(a, b, na, nb, out);
#endif
#if defined(TCLEMBEDDING_HAVE_SSE2) && defined(__x86_64__) &&                                      \
    (defined(__GNUC__) || defined(__clang__))
    if (similarity_runtime_sse2())
        return similarity_sse2_cosine(a, b, na, nb, out);
#endif
    return similarity_scalar_cosine(a, b, na, nb, out);
}

int similarity_dot_f32le(const unsigned char *a, const unsigned char *b, size_t na, size_t nb,
                         double *out) {
    return similarity_scalar_dot(a, b, na, nb, out);
}
