/* Shared similarity kernel: little-endian IEEE 754 float32 vectors.
 * Compile without fast-math. SSE2 is opt-in at compile time; the binary must
 * be deployed to CPUs that support every instruction used by the build.
 * Copyright (c) 2024
 * License: MIT
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "similarity.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "cosine_similarity requires non-finite checks; disable fast-math"
#endif
#if defined(COSINE_ENABLE_SSE2) && defined(__SSE2__) &&                                            \
    (defined(__i386__) || defined(__x86_64__)) && !defined(COSINE_FORCE_SCALAR)
#include <emmintrin.h>
#include <xmmintrin.h>
#define COSINE_SSE2 1
#else
#define COSINE_SSE2 0
#endif

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128,
               "IEEE 754 binary32 floats are required");

/* Decode bytes explicitly; no alignment or host byte-order assumptions. */
static float read_float32_le(const unsigned char *p) {
    uint32_t bits =
        (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Returns 1 for valid input and sets output; otherwise returns 0. */
int similarity_cosine_f32le(const unsigned char *a, const unsigned char *b, size_t bytes_a,
                            size_t bytes_b, double *output) {
    if (!a || !b || !output || bytes_a == 0 || bytes_a != bytes_b || bytes_a % 4 != 0)
        return 0;
    size_t n = bytes_a / 4, i = 0;
    double dot = 0.0, aa = 0.0, bb = 0.0;
#if COSINE_SSE2
    __m128d dot0 = _mm_setzero_pd(), dot1 = _mm_setzero_pd();
    __m128d aa0 = _mm_setzero_pd(), aa1 = _mm_setzero_pd();
    __m128d bb0 = _mm_setzero_pd(), bb1 = _mm_setzero_pd();
    for (; n - i >= 4; i += 4) {
        float av[4], bv[4];
        for (size_t j = 0; j < 4; j++) {
            av[j] = read_float32_le(a + 4 * (i + j));
            bv[j] = read_float32_le(b + 4 * (i + j));
            if (!isfinite(av[j]) || !isfinite(bv[j]))
                return 0;
        }
        __m128 va = _mm_loadu_ps(av), vb = _mm_loadu_ps(bv);
        __m128d a0 = _mm_cvtps_pd(va), b0 = _mm_cvtps_pd(vb);
        __m128d a1 = _mm_cvtps_pd(_mm_movehl_ps(va, va));
        __m128d b1 = _mm_cvtps_pd(_mm_movehl_ps(vb, vb));
        dot0 = _mm_add_pd(dot0, _mm_mul_pd(a0, b0));
        dot1 = _mm_add_pd(dot1, _mm_mul_pd(a1, b1));
        aa0 = _mm_add_pd(aa0, _mm_mul_pd(a0, a0));
        aa1 = _mm_add_pd(aa1, _mm_mul_pd(a1, a1));
        bb0 = _mm_add_pd(bb0, _mm_mul_pd(b0, b0));
        bb1 = _mm_add_pd(bb1, _mm_mul_pd(b1, b1));
    }
    double temp[2];
    _mm_storeu_pd(temp, _mm_add_pd(dot0, dot1));
    dot = temp[0] + temp[1];
    _mm_storeu_pd(temp, _mm_add_pd(aa0, aa1));
    aa = temp[0] + temp[1];
    _mm_storeu_pd(temp, _mm_add_pd(bb0, bb1));
    bb = temp[0] + temp[1];
#endif
    for (; i < n; i++) {
        double x = read_float32_le(a + 4 * i), y = read_float32_le(b + 4 * i);
        if (!isfinite(x) || !isfinite(y))
            return 0;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    if (!isfinite(dot) || !isfinite(aa) || !isfinite(bb) || aa <= 0.0 || bb <= 0.0)
        return 0;
    double result = (dot / sqrt(aa)) / sqrt(bb);
    if (!isfinite(result))
        return 0;
    *output = result;
    return 1;
}

int similarity_dot_f32le(const unsigned char *a, const unsigned char *b, size_t bytes_a,
                         size_t bytes_b, double *output) {
    if (!a || !b || !output || bytes_a == 0 || bytes_a != bytes_b || bytes_a % 4 != 0)
        return 0;
    double dot = 0.0;
    for (size_t i = 0; i < bytes_a / 4; i++) {
        double x = read_float32_le(a + 4 * i), y = read_float32_le(b + 4 * i);
        if (!isfinite(x) || !isfinite(y))
            return 0;
        dot += x * y;
    }
    if (!isfinite(dot))
        return 0;
    *output = dot;
    return 1;
}
