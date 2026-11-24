/* Select one path; execute it only with CPU/OS support for that path. */
#include "../src/similarity.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
int similarity_avx_cosine(const unsigned char *, const unsigned char *, size_t, size_t, double *);
int similarity_avx2_cosine(const unsigned char *, const unsigned char *, size_t, size_t, double *);
int similarity_scalar_cosine(const unsigned char *, const unsigned char *, size_t, size_t,
                             double *);
typedef int (*fn)(const unsigned char *, const unsigned char *, size_t, size_t, double *);
static void store(unsigned char *p, uint32_t bits) {
    for (int i = 0; i < 4; i++)
        p[i] = (unsigned char)(bits >> (8 * i));
}
int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "avx") && strcmp(argv[1], "avx2"))) {
        fprintf(stderr, "usage: similarity_avx_compare avx|avx2\n");
        return 2;
    }
    int selected = strcmp(argv[1], "avx2") == 0;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (selected ? !__builtin_cpu_supports("avx2") : !__builtin_cpu_supports("avx")) {
        puts("Selected CPU/OS feature is unavailable");
        return 2;
    }
#else
    return 2;
#endif
    unsigned char a[768 * 4 + 1], b[768 * 4 + 1];
    fn paths[] = {similarity_avx_cosine, similarity_avx2_cosine};
    double maxdiff = 0;
    for (size_t n = 1; n <= 768; n++) {
        for (size_t i = 0; i < n; i++) {
            float x = (float)((int)(i % 41) - 20) / 7.0f, y = (float)((int)(i % 37) - 18) / 9.0f;
            uint32_t u, v;
            memcpy(&u, &x, 4);
            memcpy(&v, &y, 4);
            store(a + 1 + 4 * i, u);
            store(b + 1 + 4 * i, v);
        }
        double reference;
        if (!similarity_scalar_cosine(a + 1, b + 1, n * 4, n * 4, &reference))
            return 1;
        for (int p = selected; p <= selected; p++) {
            double result = 9;
            if (!paths[p](a + 1, b + 1, n * 4, n * 4, &result))
                return 1;
            double delta = fabs(result - reference);
            if (delta > maxdiff)
                maxdiff = delta;
            if (delta > 1e-12)
                return 1;
        }
    }
    const uint32_t invalid[] = {0, 0x7fc00000u, 0x7f800000u, 0xff800000u};
    for (int p = selected; p <= selected; p++)
        for (int i = 0; i < 4; i++) {
            store(a + 1, invalid[i]);
            store(b + 1, 0x3f800000u);
            double out = 17;
            if (paths[p](a + 1, b + 1, 4, 4, &out) || out != 17)
                return 1;
            if (paths[p](a + 1, b + 1, 4, 3, &out) || out != 17)
                return 1;
            for (int j = 0; j < 8; j++) {
                store(a + 1 + 4 * j, invalid[i]);
                store(b + 1 + 4 * j, 0x3f800000u);
            }
            if (paths[p](a + 1, b + 1, 32, 32, &out) || out != 17)
                return 1;
        }
    for (int p = selected; p <= selected; p++) {
        for (int j = 0; j < 8; j++) {
            store(a + 1 + 4 * j, 0x7f7fffffu);
            store(b + 1 + 4 * j, 1u);
        }
        double out = 0;
        if (!paths[p](a + 1, b + 1, 32, 32, &out) || fabs(out - 1.0) > 1e-12)
            return 1;
    }
    printf("%s cases=768 max_abs=%.17g tolerance=1e-12\n", argv[1], maxdiff);
    return 0;
}
