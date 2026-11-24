/* x86-64 little-endian float32 cosine; compile this object separately. */
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "cosine_similarity requires non-finite checks; disable fast-math"
#endif
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128,
               "binary32 required");
#if defined(TCLEMBEDDING_AVX2)
#define KERNEL_NAME similarity_avx2_cosine
#else
#define KERNEL_NAME similarity_avx_cosine
#endif

int KERNEL_NAME(const unsigned char *a, const unsigned char *b, size_t na, size_t nb, double *out) {
    if (!a || !b || !out || !na || na != nb || na % 4)
        return 0;
    size_t n = na / 4, i = 0;
    __m256d dot = _mm256_setzero_pd(), aa = _mm256_setzero_pd(), bb = _mm256_setzero_pd();
    double sd = 0, sa = 0, sb = 0;
    for (; n - i >= 8; i += 8) {
        float av[8], bv[8];
        memcpy(av, a + i * 4, sizeof av);
        memcpy(bv, b + i * 4, sizeof bv);
#if defined(TCLEMBEDDING_AVX2)
        __m256i mask = _mm256_set1_epi32(0x7f800000);
        __m256i ai = _mm256_loadu_si256((const __m256i *)(const void *)av);
        __m256i bi = _mm256_loadu_si256((const __m256i *)(const void *)bv);
        __m256i bad_a = _mm256_cmpeq_epi32(_mm256_and_si256(ai, mask), mask);
        __m256i bad_b = _mm256_cmpeq_epi32(_mm256_and_si256(bi, mask), mask);
        if (_mm256_movemask_epi8(_mm256_or_si256(bad_a, bad_b)))
            return 0;
#else
        for (size_t j = 0; j < 8; j++)
            if (!isfinite(av[j]) || !isfinite(bv[j]))
                return 0;
#endif
        for (size_t j = 0; j < 8; j += 4) {
            __m256d x = _mm256_cvtps_pd(_mm_loadu_ps(av + j));
            __m256d y = _mm256_cvtps_pd(_mm_loadu_ps(bv + j));
            dot = _mm256_add_pd(dot, _mm256_mul_pd(x, y));
            aa = _mm256_add_pd(aa, _mm256_mul_pd(x, x));
            bb = _mm256_add_pd(bb, _mm256_mul_pd(y, y));
        }
    }
    double lanes[4];
    _mm256_storeu_pd(lanes, dot);
    for (int j = 0; j < 4; j++)
        sd += lanes[j];
    _mm256_storeu_pd(lanes, aa);
    for (int j = 0; j < 4; j++)
        sa += lanes[j];
    _mm256_storeu_pd(lanes, bb);
    for (int j = 0; j < 4; j++)
        sb += lanes[j];
    for (; i < n; i++) {
        float x, y;
        memcpy(&x, a + i * 4, 4);
        memcpy(&y, b + i * 4, 4);
        if (!isfinite(x) || !isfinite(y))
            return 0;
        sd += (double)x * y;
        sa += (double)x * x;
        sb += (double)y * y;
    }
    if (!isfinite(sd) || !isfinite(sa) || !isfinite(sb) || sa <= 0 || sb <= 0)
        return 0;
    double result = (sd / sqrt(sa)) / sqrt(sb);
    if (!isfinite(result))
        return 0;
    *out = result;
    return 1;
}
